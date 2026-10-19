"""Standard-library regressions for project discovery and Pi RPC lifecycle."""
import json
import os
from pathlib import Path
import sys
import signal
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from Jev_Agent.notebook_support import find_pi_extension,resolve_dsh_app,check_dsh_environment,run_pi_command

ROOT=Path(__file__).resolve().parents[2]


class PathTests(unittest.TestCase):
    def test_pi_extension_from_repository_and_notebook_directories(self):
        expected=ROOT/'Jev_Agent/pi_jev_demo/pi_jev_demo_extension.ts'
        for cwd in (ROOT,ROOT/'Jev_Agent',ROOT/'Jev_Agent/pi_jev_demo'):
            self.assertEqual(find_pi_extension(cwd),expected)

    def test_missing_dsh_app_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(FileNotFoundError,'DSH_JEV_APP'):
                resolve_dsh_app(Path(tmp))

    def test_recorded_mode_does_not_need_node_or_compiled_router(self):
        with tempfile.TemporaryDirectory() as tmp:
            app=Path(tmp);(app/'package.json').write_text('{}');(app/'validation').mkdir()
            with patch.dict(os.environ,{'DSH_JEV_APP':str(app)},clear=True):
                self.assertEqual(resolve_dsh_app(ROOT),app.resolve())
                check_dsh_environment(app,'recorded',node=None)
            with self.assertRaisesRegex(FileNotFoundError,'npm'):
                check_dsh_environment(app,'live',node=sys.executable)

    def test_invalid_explicit_override_does_not_silently_fall_back(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'DSH_JEV_APP':tmp},clear=True):
            with self.assertRaisesRegex(FileNotFoundError,'DSH_JEV_APP'):
                resolve_dsh_app(ROOT)


class RPCProcessTests(unittest.TestCase):
    def fake_pi(self,directory,body):
        script=Path(directory)/'fake-pi'
        script.write_text(f'#!{sys.executable}\n'+body)
        script.chmod(0o755)
        return str(script)

    def test_large_stderr_is_drained_and_process_reaped(self):
        with tempfile.TemporaryDirectory() as tmp:
            script=self.fake_pi(tmp,'''import json,sys,os,time
sys.stdin.readline()
sys.stderr.write('x'*200000);sys.stderr.flush()
print(json.dumps({'type':'extension_ui_request','method':'notify','message':'[pi-jev-demo] '+json.dumps({'branch':'allow'})}),flush=True)
time.sleep(20)
''')
            result=run_pi_command('test',find_pi_extension(ROOT),pi_exe=script,timeout=2)
            self.assertEqual(result['report']['branch'],'allow')

    def test_early_exit_does_not_wait_for_full_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            script=self.fake_pi(tmp,"import sys\nsys.stdin.readline()\nsys.exit(4)\n")
            start=time.monotonic()
            with self.assertRaisesRegex(RuntimeError,'exited'):
                run_pi_command('test',find_pi_extension(ROOT),pi_exe=script,timeout=10)
            self.assertLess(time.monotonic()-start,3)

    def test_timeout_cleans_up_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            pid_file=Path(tmp)/'pid'
            script=self.fake_pi(tmp,f"import os,time\nfrom pathlib import Path\nPath({str(pid_file)!r}).write_text(str(os.getpid()))\ntime.sleep(20)\n")
            with self.assertRaises(TimeoutError):
                run_pi_command('test',find_pi_extension(ROOT),pi_exe=script,timeout=.3)
            pid=int(pid_file.read_text())
            with self.assertRaises(ProcessLookupError):os.kill(pid,0)

    def test_rpc_rejection_is_reported_immediately(self):
        with tempfile.TemporaryDirectory() as tmp:
            script=self.fake_pi(tmp,"import json,sys,time\nsys.stdin.readline()\nprint(json.dumps({'type':'response','command':'prompt','success':False,'error':'failure'}),flush=True)\ntime.sleep(20)\n")
            with self.assertRaisesRegex(RuntimeError,'rejected'):
                run_pi_command('test',find_pi_extension(ROOT),pi_exe=script,timeout=1)


class NotebookIntegrationTests(unittest.TestCase):
    def cell(self, filename, index):
        notebook = json.loads((ROOT / 'Jev_Agent' / filename).read_text())
        return ''.join(notebook['cells'][index]['source'])

    def test_pi_notebook_setup_finds_extension_and_uses_bounded_runner(self):
        namespace = {}
        exec(self.cell('01_Pi集成实验.ipynb', 6), namespace)
        self.assertTrue(namespace['EXTENSION'].is_file())
        exec(self.cell('01_Pi集成实验.ipynb', 12), namespace)
        with tempfile.TemporaryDirectory() as tmp:
            fake = RPCProcessTests().fake_pi(tmp, "import sys\nsys.stdin.readline()\nsys.exit(4)\n")
            namespace['PI_EXE'] = fake
            with self.assertRaisesRegex(RuntimeError, 'exited'):
                namespace['run_pi_command']('test', timeout=.5)

    def test_dsh_notebook_recorded_setup_accepts_external_unbuilt_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = Path(tmp)
            (app / 'package.json').write_text('{}')
            (app / 'validation').mkdir()
            with patch.dict(os.environ, {'DSH_JEV_APP': tmp, 'DSH_JEV_RUN_MODE': 'recorded'}):
                namespace = {}
                exec(self.cell('02_DSH决策协作.ipynb', 2), namespace)
                self.assertEqual(namespace['APP'], app.resolve())


class DescendantCleanupTests(unittest.TestCase):
    @unittest.skipIf(os.name == 'nt', 'POSIX process groups')
    def test_pi_child_ignoring_sigterm_cannot_hold_report_cleanup_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            pid_file = Path(tmp) / 'group-pid'
            child_code = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready',flush=True); time.sleep(30)"
            body = f"""import os,sys,subprocess,json,time
from pathlib import Path
sys.stdin.readline()
Path({str(pid_file)!r}).write_text(str(os.getpid()))
child = subprocess.Popen([sys.executable, '-c', {child_code!r}], stdout=subprocess.PIPE, text=True)
child.stdout.readline()
print(json.dumps({{'type':'extension_ui_request','method':'notify','message':'[pi-jev-demo] {{}}'}}),flush=True)
time.sleep(30)
"""
            fake = RPCProcessTests().fake_pi(tmp, body)
            runner = (
                'from Jev_Agent.notebook_support import run_pi_command,find_pi_extension; '
                f'run_pi_command("test",find_pi_extension(),pi_exe={fake!r},timeout=2)'
            )
            try:
                result = subprocess.run([sys.executable, '-c', runner], cwd=ROOT,
                                        capture_output=True, text=True, timeout=6)
                self.assertEqual(result.returncode, 0, result.stderr)
            finally:
                if pid_file.exists():
                    try:
                        os.killpg(int(pid_file.read_text()), signal.SIGKILL)
                    except ProcessLookupError:
                        pass

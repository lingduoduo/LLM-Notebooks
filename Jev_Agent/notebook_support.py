"""Portable notebook discovery and bounded, cleaned-up Pi RPC subprocesses."""
from collections import deque
import json
import math
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import threading
import time

PREFIX='[pi-jev-demo] '


def find_pi_extension(start=None):
    start=Path(start or Path.cwd()).resolve()
    for root in (start,*start.parents):
        for candidate in (root/'pi_jev_demo/pi_jev_demo_extension.ts',
                          root/'Jev_Agent/pi_jev_demo/pi_jev_demo_extension.ts'):
            if candidate.is_file():return candidate
    raise FileNotFoundError('Pi extension not found; start from the repository root or Jev_Agent/')


def resolve_dsh_app(start=None):
    override=os.environ.get('DSH_JEV_APP')
    if override:
        candidates=[Path(override).expanduser().resolve()]
    else:
        start=Path(start or Path.cwd()).resolve()
        candidates=[p/'apps/dsh-jev-decision' for p in (start,*start.parents)]
    for candidate in candidates:
        if (candidate/'package.json').is_file():return candidate
    raise FileNotFoundError('DSH project is not bundled in this repository. Set DSH_JEV_APP to the external dsh-jev-decision directory containing package.json.')


def check_dsh_environment(app,mode,node=None):
    if mode not in ('live','recorded'):raise ValueError('DSH_JEV_RUN_MODE must be live or recorded')
    app=Path(app)
    if mode=='recorded':
        if not (app/'validation').is_dir():
            raise FileNotFoundError('Recorded mode requires the original validation/ archive in DSH_JEV_APP')
        return
    if not (app/'dist/router.js').is_file():
        raise FileNotFoundError('Build the external DSH project first: npm ci --ignore-scripts && npm run build')
    node=node or shutil.which('node')
    if not node:raise FileNotFoundError('Live mode requires Node.js 24 on PATH')
    if not os.environ.get('TYPESAFE_API_KEY','').strip():raise ValueError('Live mode requires TYPESAFE_API_KEY')


def _stop_process(proc):
    # The process group can outlive its leader and keep our pipes open.
    if os.name != 'nt':
        def send(sig):
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                pass
            except PermissionError:
                # macOS can report EPERM for a group containing only a zombie.
                # A still-running leader must not have its cleanup error hidden.
                if proc.poll() is None:
                    raise
        send(signal.SIGTERM)
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
        finally:
            # Escalate even when the leader exited successfully on SIGTERM.
            send(signal.SIGKILL)
        proc.wait(timeout=2)
        return
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=1)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2)


def run_pi_command(command,extension,*,pi_exe=None,timeout=30):
    if not isinstance(command,str) or not command.strip():raise ValueError('Pi command must be nonempty text')
    if type(timeout) not in (int,float) or not math.isfinite(timeout) or timeout<=0:
        raise ValueError('timeout must be positive and finite')
    extension=Path(extension).resolve()
    if not extension.is_file():raise FileNotFoundError(f'Pi extension not found: {extension}')
    executable=pi_exe or os.environ.get('PI_EXE') or ('pi.cmd' if os.name=='nt' else 'pi')
    if not (shutil.which(executable) or Path(executable).is_file()):
        raise FileNotFoundError('Pi CLI not found. Install Pi or set PI_EXE to its executable path.')
    args=[str(executable),'--mode','rpc','--no-session','--no-builtin-tools',
          '--no-skills','--no-extensions','--extension',str(extension)]
    proc=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                          text=True,encoding='utf-8',errors='replace',bufsize=1,
                          start_new_session=os.name!='nt')
    events=queue.Queue(); eof=object(); recent=deque(maxlen=3); stderr=deque(maxlen=8)
    def read_stdout():
        try:
            for line in proc.stdout:
                try:events.put(json.loads(line))
                except json.JSONDecodeError:events.put({'type':'raw','line':line.rstrip()[:500]})
        finally:events.put(eof)
    def read_stderr():
        for chunk in iter(lambda:proc.stderr.read(4096),''):stderr.append(chunk)
    readers=[threading.Thread(target=read_stdout,daemon=True),threading.Thread(target=read_stderr,daemon=True)]
    for reader in readers:reader.start()
    started=time.monotonic();deadline=started+timeout;seen=0
    try:
        try:
            proc.stdin.write(json.dumps({'type':'prompt','message':command},ensure_ascii=False)+'\n');proc.stdin.flush()
        except BrokenPipeError:raise RuntimeError('Pi exited before accepting the command') from None
        while True:
            remaining=deadline-time.monotonic()
            if remaining<=0:raise TimeoutError(f'Pi did not return a demo report within {timeout}s')
            try:event=events.get(timeout=min(.2,remaining))
            except queue.Empty:continue
            if event is eof:raise RuntimeError('Pi exited without returning a demo report')
            seen+=1;recent.append(event)
            if not isinstance(event,dict):continue
            if event.get('type')=='response' and event.get('success') is False:
                raise RuntimeError('Pi rejected the RPC command')
            if event.get('type')=='extension_ui_request' and event.get('method')=='notify':
                message=event.get('message','')
                if isinstance(message,str) and message.startswith(PREFIX):
                    report=json.loads(message[len(PREFIX):])
                    if not isinstance(report,dict):raise ValueError('Invalid Pi demo report')
                    return {'elapsed_ms':round((time.monotonic()-started)*1000,1),
                            'report':report,'events_seen':seen}
    finally:
        _stop_process(proc)
        for reader in readers:reader.join(timeout=1)
        for stream in (proc.stdin,proc.stdout,proc.stderr):stream.close()

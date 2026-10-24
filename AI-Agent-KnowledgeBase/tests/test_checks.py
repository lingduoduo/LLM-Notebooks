import unittest

from run_checks import ROOT, check_environment, suite_commands


class CheckRunnerTests(unittest.TestCase):
    def test_suites_are_separate_processes_with_local_working_directories(self):
        suites = suite_commands("/test/python")
        self.assertEqual(len(suites), 8)
        for name, (directory, command) in suites.items():
            with self.subTest(name=name):
                self.assertTrue(directory.is_relative_to(ROOT))
                self.assertEqual(command[:3], ["/test/python", "-m", "pytest"])
                self.assertTrue(
                    any(arg.startswith("test_") or arg == "tests" for arg in command)
                )
        pipeline_args = suites["retrieval-pipeline"][1]
        for live_script in (
            "test_client.py",
            "test_pipeline.py",
            "test_improvements.py",
        ):
            self.assertNotIn(live_script, pipeline_args)
        self.assertIn("test_fusion.py", pipeline_args)

    def test_offline_environment_overrides_live_opt_ins_without_modifying_parent(self):
        source = {"RUN_LIVE_SMOKE_TESTS": "1", "STRUCTURED_INDEX_E2E": "1"}
        environment = check_environment(source)
        self.assertEqual(environment["RUN_LIVE_SMOKE_TESTS"], "0")
        self.assertEqual(environment["STRUCTURED_INDEX_E2E"], "")
        self.assertEqual(environment["HF_HUB_OFFLINE"], "1")
        self.assertEqual(source["RUN_LIVE_SMOKE_TESTS"], "1")

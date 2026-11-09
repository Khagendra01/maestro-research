import json
import tempfile
import unittest
from pathlib import Path

from maestro_research.backends.dryrun import DryRunBackend
from maestro_research.episode_runner import run_episode, write_summary
from maestro_research.manifest import load_manifest
from maestro_research.metrics import summarize_records
from maestro_research.skills import ExecutorConfig, StagedExecutor

MANIFEST = Path(__file__).resolve().parent.parent / "maestro_research" / "configs" / "dev_smoke_r2.json"


class ManifestTests(unittest.TestCase):
    def test_manifest_is_pinned_and_expands_to_fifteen_episodes(self):
        manifest = load_manifest(MANIFEST)
        self.assertEqual(len(manifest.tasks), 3)
        self.assertEqual(len({task.suite for task in manifest.tasks}), 3)
        episodes = manifest.episodes()
        self.assertEqual(len(episodes), 15)
        self.assertEqual(len({episode.episode_id for episode in episodes}), 15)
        self.assertEqual(manifest.max_steps("libero_object"), 280)

    def test_limit_per_task(self):
        manifest = load_manifest(MANIFEST)
        self.assertEqual(len(manifest.episodes(limit_per_task=1)), 3)


class DryRunPipelineTests(unittest.TestCase):
    def test_end_to_end_executor_reaches_scene_success(self):
        manifest = load_manifest(MANIFEST)
        backend = DryRunBackend()
        executor = StagedExecutor(ExecutorConfig(grasp_retries=1))
        with tempfile.TemporaryDirectory() as tmp:
            record = run_episode(
                backend,
                manifest.episodes(limit_per_task=1)[0],
                manifest,
                executor,
                tmp,
                settings="oracle",
                save_frames=False,
            )
            self.assertEqual(record["status"], "ok")
            self.assertTrue(record["official_success"])
            summary = write_summary(tmp, [record])
            data = json.loads(Path(summary).read_text())
            self.assertEqual(data["successes"], 1)

    def test_summary_groups_by_suite(self):
        summary = summarize_records(
            [
                {"suite": "libero_object", "official_success": True},
                {"suite": "libero_object", "official_success": False},
                {"suite": "libero_goal", "official_success": True},
            ]
        )
        self.assertEqual(summary["episodes"], 3)
        self.assertEqual(summary["by_group"]["libero_object"]["successes"], 1)
        self.assertAlmostEqual(summary["by_group"]["libero_goal"]["success_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()

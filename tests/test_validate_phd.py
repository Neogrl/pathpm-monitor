import tempfile
import unittest
from pathlib import Path

from tools.validate_phd import (
    generate_replay_sequence,
    replay_phd,
    scenario_config,
    write_validation_outputs,
)


class PHDValidationTests(unittest.TestCase):
    def test_overlap_replay_uses_one_team_detection_and_writes_outputs(self) -> None:
        cfg = scenario_config("overlap", 1)
        frames = generate_replay_sequence(cfg, "overlap", seed=20, steps=4)
        self.assertTrue(all(frame.true_detection_count == 1 for frame in frames))
        self.assertTrue(all(len(frame.measurements) == 1 for frame in frames))

        trace, support, summary = replay_phd(
            cfg, frames, particle_count=600, belief_seed=20
        )
        self.assertEqual(len(trace), 4)
        self.assertEqual(summary["steps"], 4.0)
        self.assertTrue(all(row["true_detection_count"] == 1 for row in trace))

        with tempfile.TemporaryDirectory() as temp_dir:
            out = Path(temp_dir)
            write_validation_outputs(out, cfg, frames, trace, support, summary)
            for name in (
                "per_step.csv",
                "particle_support.csv",
                "summary.json",
                "replay_sequence.npz",
                "count_curve.png",
                "position_error_curve.png",
                "ess_curve.png",
                "phd_mass_curve.png",
            ):
                self.assertTrue((out / name).exists(), name)


if __name__ == "__main__":
    unittest.main()

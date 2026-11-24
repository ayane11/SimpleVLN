import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from navigation_memory import (  # noqa: E402
    action_plan_sentence,
    clean_reasoning_update,
    clean_stale_progress_tail,
    rewrite_next_plan,
    strip_stale_next_plan,
)
from clean_cot_index import clean_reasoning_root, clean_segments  # noqa: E402


class CleanCotIndexTests(unittest.TestCase):
    def test_clean_stale_progress_tail_drops_duplicate_action_sentence(self):
        raw = (
            "I see a hallway with light walls. "
            "The agent has exited the bedroom and is currently in the hallway. "
            "The next step is to turn left to proceed towards the living area. "
            "The next plan is to turn left."
        )
        cleaned = clean_reasoning_update(raw)
        self.assertEqual(
            cleaned,
            "I see a hallway with light walls. "
            "The agent has exited the bedroom and is currently in the hallway. "
            "The next plan is to turn left.",
        )
        # Idempotent
        self.assertEqual(clean_reasoning_update(cleaned), cleaned)

    def test_clean_stale_progress_tail_trims_compound_clause_and_keeps_negations(self):
        compound = (
            "I see a large mirror. "
            "The agent has reached the top of the stairs, and the agent is now positioned to take the first left turn into the hall."
        )
        self.assertEqual(
            clean_stale_progress_tail(compound),
            "I see a large mirror. The agent has reached the top of the stairs.",
        )
        negated = (
            "I see a corridor. "
            "The agent has entered the hall. "
            "The instruction to turn left into the bedroom has not yet been executed."
        )
        self.assertEqual(clean_stale_progress_tail(negated), negated)

    def test_rewrite_next_plan_and_strip_stale_plan(self):
        raw = (
            "I see a tiled room with wooden lockers on the right. "
            "The agent has entered the locker room. "
            "The agent is currently facing the lattice screen and needs to turn right to face the lockers. "
            "The next plan is to turn right."
        )
        rewritten = rewrite_next_plan(raw, "forward 25 cm")
        self.assertEqual(
            rewritten,
            "I see a tiled room with wooden lockers on the right. "
            "The agent has entered the locker room. "
            "The next plan is to move forward.",
        )
        self.assertEqual(
            strip_stale_next_plan(raw),
            "I see a tiled room with wooden lockers on the right. "
            "The agent has entered the locker room.",
        )
        self.assertEqual(action_plan_sentence("stop"), "The next plan is to stop and wait.")
        self.assertEqual(rewrite_next_plan("Entered kitchen.", "turn left 15 degree"), "Entered kitchen.")

    def test_clean_segments_propagates_updated_content_to_following_content(self):
        raw_update = (
            "I see a lobby with a piano. "
            "The agent has moved into the lobby. "
            "The agent is positioned to turn left towards the kitchen area. "
            "The next plan is to turn left."
        )
        segments = [
            {"start_step": 0, "end_step": 10, "content": None, "updated_content": None, "type": "only_act"},
            {"start_step": 10, "end_step": 15, "content": None, "updated_content": raw_update, "type": "newroom", "index": 10},
            {"start_step": 15, "end_step": -1, "content": raw_update, "updated_content": None, "type": "only_act"},
        ]
        cleaned_segs, n_tot, n_chg = clean_segments(segments)
        self.assertEqual((n_tot, n_chg), (1, 1))
        expected = "I see a lobby with a piano. The agent has moved into the lobby. The next plan is to turn left."
        self.assertEqual(cleaned_segs[1]["updated_content"], expected)
        self.assertEqual(cleaned_segs[2]["content"], expected)

    def test_clean_reasoning_root_creates_backup_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen_r2r = root / "generation" / "r2r"
            gen_r2r.mkdir(parents=True)
            raw_text = (
                "I see a hallway. The agent has exited the room. "
                "The next step is to turn right. The next plan is to turn right."
            )
            payload = {
                '["r2r","1",0]': {
                    "episode_id": "1",
                    "dataset": "r2r",
                    "instruction_index": 0,
                    "instruction": "go",
                    "review_status": "pending",
                    "segments": [
                        {"start_step": 5, "end_step": 10, "content": None, "updated_content": raw_text, "type": "newroom", "index": 5},
                    ],
                }
            }
            (gen_r2r / "cot_index.json").write_text(json.dumps(payload))
            summary1 = clean_reasoning_root(root)
            self.assertEqual(summary1["r2r"]["cleaned_nodes"], 1)
            self.assertTrue((gen_r2r / "cot_index.uncleaned.json").is_file())
            exported = json.loads((root / "exports" / "cot_index.json").read_text())
            self.assertNotIn("The next step is to turn right.", exported['["r2r","1",0]']["segments"][0]["updated_content"])
            # Second run still reads from .uncleaned.json and produces identical output
            summary2 = clean_reasoning_root(root)
            self.assertEqual(summary2["r2r"]["cleaned_nodes"], 1)


if __name__ == "__main__":
    unittest.main()

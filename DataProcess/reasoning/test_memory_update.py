import unittest
from core import memory_steps, validate_memory_reason, reason_text


class MemoryUpdateTests(unittest.TestCase):
    def value(self, **updates):
        v = dict(scene_description='I see a doorway.',
                 progress_assessment='I have approached the doorway.',
                 next_plan='The next plan is to move forward.',
                 evidence_steps=[100], uncertainties=[])
        v.update(updates)
        return v

    def test_causal_bounded_recent_and_memory_anchor(self):
        for last in range(130):
            steps = memory_steps(last, 8, 30 if last > 30 else None)
            self.assertLessEqual(len(steps), 8)
            self.assertEqual(steps, sorted(set(steps)))
            self.assertTrue(all(0 <= s <= last for s in steps))
            self.assertEqual(steps[-min(5, last+1):], list(range(max(0,last-4),last+1)))
        self.assertIn(30, memory_steps(100, 8, 30))

    def test_stale_whole_memory_rejected(self):
        v = self.value()
        with self.assertRaisesRegex(ValueError, 'old memory'):
            validate_memory_reason(v, [100], ['forward 25 cm'], reason_text(v))

    def test_current_evidence_required(self):
        with self.assertRaisesRegex(ValueError, 'CURRENT'):
            validate_memory_reason(self.value(evidence_steps=[90]), [90,100], ['forward 25 cm'])

    def test_immediate_stop_and_global_completion(self):
        for updates in [dict(next_plan='The next plan is to stop here.'),
                        dict(progress_assessment='I have completed all the instructions.')]:
            with self.assertRaises(ValueError):
                validate_memory_reason(self.value(**updates), [100], ['forward 25 cm'])

    def test_local_completion_and_conditional_goal_not_rejected(self):
        validate_memory_reason(self.value(progress_assessment='I have completed the turn.',
            next_plan='Continue toward the door, then stop at the destination.'), [100], ['forward 25 cm'])

    def test_expert_stop_allowed(self):
        validate_memory_reason(self.value(progress_assessment='I have reached the final destination.',
            next_plan='The next plan is to stop here.'), [100], ['stop'])


if __name__ == '__main__':
    unittest.main()

import unittest
from room_events import detect, export, category


def rows(names):
    return [dict(step=i, region_candidates=[] if n is None else
                 [dict(id=str(i), category=n)]) for i, n in enumerate(names)]


class RoomEventsTest(unittest.TestCase):
    def test_confirm_without_backdating(self):
        self.assertEqual([n['step'] for n in detect(rows(['a','a','b','b']))], [3])

    def test_jitter(self):
        self.assertEqual(detect(rows(['a','a','b','a','b','a'])), [])

    def test_unknown_breaks_confirmation(self):
        result = detect(rows(['a','a','b',None,'b','b']))
        self.assertEqual(result[0]['step'], 5)
        self.assertTrue(result[0]['crossed_unknown'])

    def test_return(self):
        self.assertEqual([n['room_to'] for n in detect(rows(['a','a','b','b','a','a']))], ['b','a'])

    def test_same_category_different_ids(self):
        self.assertEqual(detect(rows(['hallway'] * 6)), [])

    def test_ambiguous(self):
        self.assertIsNone(category([{'category':'a'}, {'category':'b'}]))
        self.assertEqual(category([{'category':'a'}, {'category':'a'}]), 'a')
        self.assertEqual(detect(rows([None] * 4)), [])

    def test_alignment_gate(self):
        self.assertEqual(export([{'episode_id':1, 'steps':rows(['a','a','b','b'])}])['events'], {})

    def test_expert_turn_conflict(self):
        from core import validate_expert_reason
        v = dict(scene_description='A door on my right.', progress_assessment='I am approaching.',
                 next_plan='Turn left.', evidence_steps=[0], uncertainties=[])
        with self.assertRaises(ValueError):
            validate_expert_reason(v, [0], ['turn right 15 degree'])
        v['next_plan'] = 'Approach the door on the right.'
        validate_expert_reason(v, [0], ['forward 25 cm'])

    def test_zero_nodes_no_teacher_call(self):
        from generate import annotate
        from types import SimpleNamespace
        from pathlib import Path
        class NoTeacher:
            def generate(self, messages):
                raise AssertionError('Zero events must not call teacher')
        plan = dict(images=['unused.jpg'], supplied_nodes=[], global_message={},
                    instruction='Stop', actions=['stop'])
        result = annotate(plan, NoTeacher(), SimpleNamespace(), {}, Path('/unused'))
        self.assertEqual(result['nodes'], [])
        self.assertIsNone(result['global_request_id'])
        self.assertFalse(result['teacher_has_future_context'])


if __name__ == '__main__':
    unittest.main()

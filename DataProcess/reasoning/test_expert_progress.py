import unittest
from expert_progress import validate_progress, validate_subtasks, action_sentence, action_runs

class ExpertProgressTest(unittest.TestCase):
    def value(self, action='forward 25 cm', **updates):
        v=dict(scene_description='I see a doorway.',progress_assessment='I am approaching the doorway.',
               next_plan=action_sentence(action),evidence_steps=[10],uncertainties=[],
               immediate_action='stop' if action=='stop' else 'forward',stop_now=action=='stop',task_complete=False,
               progress_state=[dict(id=1,status='in_progress',evidence_steps=[10])],active_subtask=1,corrections=[])
        v.update(updates);return v
    def test_nearby_stop_never_permits_immediate_stop(self):
        for updates in [dict(stop_now=True),dict(task_complete=True),dict(next_plan='The next plan is to stop immediately.')]:
            with self.assertRaises(ValueError):validate_progress(self.value(**updates),[10],'forward 25 cm',['Enter.'])
    def test_literal_plan_must_match_structured_action(self):
        with self.assertRaises(ValueError):validate_progress(self.value(next_plan='The next plan is to make a slight right turn.'),[10],'forward 25 cm',['Enter.'])
    def test_completed_memory_can_retain_old_evidence(self):
        state=[dict(id=1,status='completed',evidence_steps=[2])]
        validate_progress(self.value(progress_state=state,active_subtask=None),[9,10],'forward 25 cm',['Enter.'],state)
    def test_cannot_invent_completion_using_unseen_frame(self):
        with self.assertRaises(ValueError):validate_progress(self.value(progress_state=[dict(id=1,status='completed',evidence_steps=[2])],active_subtask=None),[10],'forward 25 cm',['Enter.'])
    def test_completed_state_correction_needs_fresh_evidence(self):
        old=[dict(id=1,status='completed',evidence_steps=[2])]
        with self.assertRaises(ValueError):validate_progress(self.value(),[10],'forward 25 cm',['Enter.'],old)
        validate_progress(self.value(corrections=[dict(id=1,reason='Current doorway shows old completion was premature.',evidence_steps=[10])]),[10],'forward 25 cm',['Enter.'],old)
    def test_stop_is_not_proof_of_semantic_completion(self):
        validate_progress(self.value('stop'),[10],'stop',['Enter.'])
        with self.assertRaises(ValueError):validate_progress(self.value('stop',task_complete=True),[10],'stop',['Enter.'])
    def test_terminal_stop_subtask_cannot_be_completed_early(self):
        v=self.value(progress_state=[dict(id=1,status='completed',evidence_steps=[10])],active_subtask=None)
        with self.assertRaisesRegex(ValueError,'Terminal stop'):
            validate_progress(v,[10],'forward 25 cm',['Stop just after passing the doorway.'])
    def test_intermediate_pause_is_not_terminal_stop(self):
        from expert_progress import terminal_stop_ids
        self.assertEqual(terminal_stop_ids(['Stop by the fireplace.','Turn right and walk to the bed.']),[])
        self.assertEqual(terminal_stop_ids(['Walk upstairs.','Wait by the door.']),[2])
    def test_negation_alone_is_not_a_validation_error(self):
        validate_progress(self.value(progress_assessment='I have not yet stopped.'),[10],'forward 25 cm',['Enter.'])
    def test_completion_can_combine_retained_and_new_evidence(self):
        old=[dict(id=1,status='in_progress',evidence_steps=[2])]
        v=self.value(progress_state=[dict(id=1,status='completed',evidence_steps=[2,10])],active_subtask=None)
        validate_progress(v,[10],'forward 25 cm',['Enter.'],old)
        v['progress_state'][0]['evidence_steps']=[2]
        with self.assertRaises(ValueError):validate_progress(v,[10],'forward 25 cm',['Enter.'],old)
    def test_no_privileged_target_references(self):
        with self.assertRaises(ValueError):validate_progress(self.value(scene_description='Following expert hints, I see a door.'),[10],'forward 25 cm',['Enter.'])
    def test_decomposition_preserves_words(self):
        validate_subtasks({'subtasks':['Go left.','Then stop.']},'Go left. Then stop.')
        for spans in [['Go left.'],['Go right. Then stop.'],['Then stop. Go left.']]:
            with self.assertRaises(ValueError):validate_subtasks({'subtasks':spans},'Go left. Then stop.')
    def test_action_runs_preserve_order(self):
        self.assertEqual(action_runs(['forward','forward','stop']),[{'action':'forward','count':2},{'action':'stop','count':1}])

if __name__=='__main__':unittest.main()

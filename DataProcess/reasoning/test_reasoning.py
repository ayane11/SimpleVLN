"""CPU tests for indexing, causal evidence, resume, and real converter integration."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
import threading
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import (sample_steps, parse_json, validate_reason, validate_nodes, make_segments)
from generate import plan_episode, annotate, export_cot, atomic, run
from teacher import check_weights, processor_messages


def reason(step):
    return dict(scene_description='I see a doorway.', progress_assessment='I have reached the doorway.',
                next_plan='The next plan is to enter the room.', evidence_steps=[step], uncertainties=[])


class FakeTeacher:
    def __init__(self):
        self.calls = 0

    def generate(self, messages):
        self.calls += 1
        if len(messages) == 2:
            v = dict(episode_summary='Follow the corridor into the room.', nodes=[dict(step=1, type='newroom')])
        else:
            v = reason(1)
        return dict(text=json.dumps(v), input_tokens=10, output_tokens=10, seconds=0)


class ReasoningTests(unittest.TestCase):
    def test_multimodal_processor_normalizes_all_roles_without_mutation(self):
        source = [{'role': 'system', 'content': 'policy'}, {'role': 'assistant', 'content': '{}'},
                  {'role': 'user', 'content': [{'type': 'image', 'image': '/tmp/example.jpg'}]}]
        normalized = processor_messages(source)
        self.assertEqual(normalized[0]['content'], [{'type': 'text', 'text': 'policy'}])
        self.assertEqual(normalized[1]['content'], [{'type': 'text', 'text': '{}'}])
        self.assertEqual(source[0]['content'], 'policy')

    def test_sampling_keeps_current_without_future(self):
        for last in range(30):
            steps = sample_steps(last, 8)
            self.assertEqual(steps[-1], last)
            self.assertLessEqual(len(steps), 8)
            self.assertTrue(all(0 <= x <= last for x in steps))
        self.assertEqual(sample_steps(3, 0), [0, 1, 2, 3])

    def test_json_does_not_take_hidden_thinking_as_label(self):
        self.assertEqual(parse_json('<think>{"wrong":1}</think>```json\n{"ok":2}\n```'), {'ok': 2})
        with self.assertRaises(ValueError):
            parse_json('{"incomplete":')

    def test_evidence_rejects_future_and_unsupplied_frames(self):
        with self.assertRaises(ValueError):
            validate_reason(reason(8), [0, 2, 4])
        with self.assertRaises(ValueError):
            validate_reason(reason(1), [0, 2, 4])
        validate_reason(reason(4), [0, 2, 4])

    def test_proposal_cannot_invent_correction(self):
        with self.assertRaises(ValueError):
            validate_nodes([dict(step=2, type='correction_part1')], 5, proposed=True, visible_steps=[2])
        with self.assertRaises(ValueError):
            validate_nodes([dict(step=1, type='newroom')], 5, proposed=True, visible_steps=[2])

    def test_events_require_valid_correction_evidence_and_bounds(self):
        with self.assertRaises(ValueError):
            validate_nodes([dict(step=2, type='correction_part1')], 5)
        with self.assertRaises(ValueError):
            validate_nodes([dict(step=2, type='newroom', progress_ratio=float('nan'))], 5)
        with self.assertRaises(ValueError):
            validate_nodes([dict(step=2, type='newroom'), dict(step=2, type='newroom')], 5)

    def test_segments_never_propagate_new_reasoning_backwards(self):
        nodes = [dict(step=2, type='newroom', reasoning=reason(2)),
                 dict(step=5, type='newroom', reasoning=reason(5))]
        segments = make_segments(nodes, 7)
        for t in range(7):
            matches = [s for s in segments if s['start_step'] <= t and (s['end_step'] == -1 or t < s['end_step'])]
            self.assertEqual(len(matches), 1)
            if t < 2:
                self.assertIsNone(matches[0]['content'])
                self.assertIsNone(matches[0]['updated_content'])
            if t in (2, 5):
                self.assertIsNotNone(matches[0]['updated_content'])
        self.assertEqual(make_segments([], 3)[0]['type'], 'only_act')

    def test_reasoning_windows_match_aware_without_overlap_or_future_leakage(self):
        nodes = [dict(step=4, type='newroom', reasoning=reason(4)),
                 dict(step=7, type='newroom', reasoning=reason(7))]
        segments = make_segments(nodes, 20)
        updates = [s for s in segments if s['updated_content']]
        self.assertEqual([(s['start_step'], s['end_step']) for s in updates], [(4, 7), (7, 12)])
        self.assertEqual(segments[-1]['content'], updates[-1]['updated_content'])
        self.assertIsNone(segments[0]['content'])
        for step in range(20):
            self.assertEqual(sum(s['start_step'] <= step and (s['end_step'] == -1 or step < s['end_step'])
                                 for s in segments), 1)
            active = next(s for s in segments if s['start_step'] <= step and (s['end_step'] == -1 or step < s['end_step']))
            ref_end = active['reference_end_step'] if active['reference_end_step'] >= 0 else 1000
            self.assertLess(active['reference_start_step'], min(ref_end, step + 1),
                            'Aware reader randrange must not receive an empty reference interval')

    def test_unlimited_proposals_do_not_impose_density_quota(self):
        nodes = [dict(step=i, type='newroom') for i in range(8)]
        self.assertEqual(len(validate_nodes(nodes, 8, proposed=True, visible_steps=list(range(8)), max_nodes=0)), 8)

    def test_concurrent_generation_resume_and_endpoint_assignment(self):
        class SlowTeacher(FakeTeacher):
            active, peak, endpoints = 0, 0, []
            lock = threading.Lock()
            def __init__(self, args):
                super().__init__()
                self.endpoints.append(args.base_url)
            def generate(self, messages):
                with self.lock:
                    type(self).active += 1
                    type(self).peak = max(self.peak, self.active)
                time.sleep(0.02)
                try:
                    return super().generate(messages)
                finally:
                    with self.lock:
                        type(self).active -= 1
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); episodes = []
            for eid in range(4):
                images = root/f'scene/{eid}/rgb'; images.mkdir(parents=True)
                for step in range(1, 4):
                    (images/f'{step:03}.jpg').write_bytes(b'fake image for fake teacher')
                episodes.append(dict(id=str(eid), video=f'scene/{eid}', actions=[-1,2,1], instructions=['Enter.']))
            source = root/'annotations.json'; atomic(source, episodes)
            out = root/'output'; out.mkdir()
            args = argparse.Namespace(annotations=str(source), image_root=str(root), dataset='r2r',
                events_json=None, model=str(root/'absent-model'), backend='vllm', base_url='http://127.0.0.1:1/v1',
                base_urls=['http://127.0.0.1:1/v1','http://127.0.0.1:2/v1'], served_model='fake',
                max_pixels=262144, max_new_tokens=1536, max_context=32768, global_frames=0,
                history_frames=8, correction_frames=8, max_nodes=0, gpu_memory_gib=40, dry_run=False,
                episode_id=None, sample_size=0, seed=1, limit=0, workers=4, retries=0)
            with patch('generate.VLLMTeacher', SlowTeacher):
                run(args, out)
                self.assertGreater(SlowTeacher.peak, 1)
                self.assertEqual(set(SlowTeacher.endpoints), set(args.base_urls))
                self.assertEqual(len(list((out/'episodes').glob('*.json'))), 4)
                before = len(SlowTeacher.endpoints)
                run(args, out)
                self.assertEqual(len(SlowTeacher.endpoints), before)
                self.assertTrue(all(r['status']=='cached' for r in json.loads((out/'run_summary.json').read_text())['results']))
                # Incomplete episode must be retried; other episodes stay cached.
                target = next((out/'episodes').glob('*.json'))
                old = json.loads(target.read_text())
                old['failed_nodes'] = [dict(step=1, error='transient')]
                atomic(target, old)
                args.retry_failed = True
                with patch('generate.annotate', wraps=annotate) as spy:
                    run(args, out)
                self.assertEqual(spy.call_count, 1)
                self.assertEqual(json.loads(target.read_text())['failed_nodes'], [])

    def test_excluded_node_does_not_call_teacher_or_update_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = dict(dataset='r2r', episode_id='1', images=['unused.jpg']*3,
                        supplied_nodes=[dict(step=1, type='newroom')], actions=['forward 25 cm']*3)
            args = argparse.Namespace(node_exclusions={'r2r': {'1': {'1': 'bad progress'}}}, retries=0)
            class NoTeacher:
                def generate(self, messages):
                    raise AssertionError('Excluded node must not call teacher')
            result = annotate(plan, NoTeacher(), args, {}, Path(tmp))
            self.assertEqual(result['nodes'], [])
            self.assertEqual(result['excluded_nodes'], [dict(step=1, reason='bad progress')])
            self.assertEqual(result['failed_nodes'], [])
            self.assertTrue(all(x.get('content') is None for x in result['segments']))

    def test_missing_download_shard_fails_before_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p/'config.json').write_text('{}')
            (p/'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'x':'missing.safetensors'}}))
            with self.assertRaisesRegex(ValueError, 'download incomplete'):
                check_weights(p)

    def test_pipeline_alignment_caching_and_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); frames = root/'scene'/'000705'/'rgb'; frames.mkdir(parents=True)
            # Converter checks paths/indices; fake backend deliberately never decodes images.
            for n in range(1, 4):
                (frames/f'{n:03}.jpg').write_bytes(b'test fixture, not a real image')
            ep = dict(id=705, video='scene/000705', actions=[-1, 2, 1], instructions=['Enter the room.', 'Go inside.'])
            args = argparse.Namespace(dataset='r2r', global_frames=8, max_nodes=5, history_frames=8,
                                      correction_frames=8, retries=0)
            plan = plan_episode(ep, 0, root, args, None)
            self.assertEqual(plan['actions'], ['turn left 15 degree', 'forward 25 cm', 'stop'])
            self.assertEqual(len(plan['images']), 3)
            self.assertTrue(plan['images'][0].endswith('001.jpg'))
            self.assertNotEqual(plan['key'], plan_episode(ep, 1, root, args, None)['key'])
            with self.assertRaisesRegex(ValueError, 'Missing events'):
                plan_episode(ep, 0, root, args, {})
            teacher = FakeTeacher(); output = root/'out'
            record = annotate(plan, teacher, args, {'test': True}, output)
            self.assertEqual(teacher.calls, 2)
            annotate(plan, teacher, args, {'test': True}, output)
            self.assertEqual(teacher.calls, 2, 'completed requests should resume without model calls')
            self.assertEqual(record['review_status'], 'pending')
            atomic(output/'episodes'/'fixture.json', record)
            self.assertEqual(export_cot(output), 1)
            self.assertIn(plan['key'], json.loads((output/'cot_index.json').read_text()))
            self.assertEqual(set(json.loads((output/'cot_new.json').read_text())), {'705'})


if __name__ == '__main__':
    unittest.main()

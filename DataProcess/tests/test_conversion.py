"""Regression tests for navigation sample conversion and supported CLI options."""

import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT / 'tool'))
from navigation_memory import strip_stale_next_plan
import convert_common as common
import convert_vln as vln
import convert_dagger as dagger


class ConversionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(setattr, vln, 'MAX_HISTORY', 8)
        self.addCleanup(setattr, dagger, 'MAX_HISTORY', 8)

    def frames(self, video, count, rgb=False):
        folder = self.root / video / ('rgb' if rgb else '')
        folder.mkdir(parents=True)
        for i in range(count):
            (folder / f'{i:03d}.jpg').touch()
        return folder

    def episode(self, **kwargs):
        episode = dict(episode_id=1, video_id='scene_1', instruction=['go'],
                       actions=[-1, 1, 2, 3])
        episode.update(kwargs)
        return episode

    def test_nested_paths_and_both_image_roots(self):
        self.frames('images/nested/scene_1', 4, rgb=True)
        ep = self.episode(video='images/nested/scene_1', instructions=['go'])
        a, b = [], []
        random.seed(1); vln.convert_episode_vln(ep, self.root, a)
        random.seed(1); vln.convert_episode_vln(ep, self.root / 'images', b)
        self.assertEqual(a, b)
        self.assertTrue(a)
        c = []
        random.seed(1); dagger.convert_episode_atomic(ep, self.root, c)
        self.assertEqual(a, c)
        for sample in a:
            self.assertEqual(sample['messages'][0]['value'].count('<image>'), len(sample['images']))

    def test_json_whitespace_bom_jsonl_and_invalid_array(self):
        path = self.root / 'input.json'
        for prefix in ('', '\n  ', '\ufeff\n '):
            path.write_text(prefix + json.dumps([self.episode()], indent=2))
            self.assertEqual(common.load_episodes(path), [self.episode()])
        path.write_text('\ufeff\n' + json.dumps(self.episode()) + '\n' + json.dumps(self.episode()))
        self.assertEqual(len(common.load_episodes(path)), 2)
        path.write_text('[1,2]')
        with self.assertRaises(ValueError): common.load_episodes(path)

    def test_node_boundary_and_memory_supervision(self):
        self.frames('scene_1', 4)
        ep = self.episode()
        index = {'["r2r","1",0]': {'instruction': 'go', 'segments': [
            {'type': 'newroom', 'index': 1, 'updated_content': 'door visible'},
            {'type': 'newroom', 'index': 3, 'updated_content': 'at goal'}]}}
        samples, control = [], []
        for rows, plain in [(samples, False), (control, True)]:
            vln.convert_episode_vln(ep, self.root, rows, cot_index=index, dataset='r2r', action_control=plain)
        actions = [row for row in samples if row['messages'][1]['value'].startswith('ACTION: ')]
        self.assertEqual([r['messages'][1]['value'][8:] for r in actions],
                         [r['messages'][1]['value'] for r in control])
        self.assertEqual([Path(r['images'][-1]).name for r in samples],
                         ['000.jpg', '001.jpg', '001.jpg', '003.jpg', '003.jpg'])
        self.assertNotIn('door visible', samples[1]['messages'][0]['value'])
        self.assertIn('Navigation memory (updated 0 steps ago): door visible', samples[2]['messages'][0]['value'])
        self.assertIn('Navigation memory (updated 2 steps ago): door visible', samples[3]['messages'][0]['value'])
        self.assertNotIn('at goal', samples[3]['messages'][0]['value'])
        self.assertIn('Navigation memory (updated 0 steps ago): at goal', samples[4]['messages'][0]['value'])

    def test_steps_ago_and_stale_next_plan_stripping(self):
        self.frames('scene_1', 6)
        ep = self.episode(actions=[-1, 1, 2, 1, 3, 1, 0])
        index = {'["r2r","1",0]': {'instruction': 'go', 'segments': [
            {'type': 'newroom', 'index': 0, 'updated_content': 'I see a hall. The next plan is to move forward.'},
            {'type': 'newroom', 'index': 5, 'updated_content': 'I see a door. The next plan is to stop and wait.'}]}}
        samples = []
        with patch.object(vln, 'MAX_ACTIONS_PER_SAMPLE', 1), patch.object(vln, 'COMBINE_PROB', 0.0):
            vln.convert_episode_vln(ep, self.root, samples, cot_index=index, dataset='r2r')
        self.assertIn('updated 0 steps ago): I see a hall. The next plan is to move forward.', samples[1]['messages'][0]['value'])
        self.assertIn('updated 1 step ago): I see a hall.', samples[2]['messages'][0]['value'])
        self.assertNotIn('The next plan is to move forward.', samples[2]['messages'][0]['value'])
        self.assertIn('updated 5 steps ago): I see a hall.', samples[6]['messages'][0]['value'])
        self.assertEqual(strip_stale_next_plan('The next plan is to move forward.'), 'The next plan is to move forward.')

    def test_scheme_1a_sampling_buffer_drop_and_stop_protection(self):
        self.frames('scene_1', 15)
        ep = self.episode(actions=[-1] + [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1, 0])
        index = {'["r2r","1",0]': {'instruction': 'go', 'segments': [
            {'type': 'newroom', 'index': 12, 'updated_content': 'I see stairs. The next plan is to move forward.'}]}}
        full_samples, filtered_samples = [], []
        with patch.object(vln, 'MAX_ACTIONS_PER_SAMPLE', 1), patch.object(vln, 'COMBINE_PROB', 0.0):
            vln.convert_episode_vln(ep, self.root, full_samples, cot_index=index, dataset='r2r')
            vln.convert_episode_vln(
                ep, self.root, filtered_samples, cot_index=index, dataset='r2r',
                drop_pre_node_buffer=True, direct_action_keep_prob=0.35, stop_repeat=2,
                sampling_rng=random.Random(42))
        self.assertFalse(any(Path(r['images'][-1]).name == '011.jpg' for r in filtered_samples))
        full_reasons = [r for r in full_samples if r['messages'][1]['value'].startswith('REASON: ')]
        filt_reasons = [r for r in filtered_samples if r['messages'][1]['value'].startswith('REASON: ')]
        self.assertEqual(filt_reasons, full_reasons)
        filt_after = [r for r in filtered_samples if 'Return ACTION:' in r['messages'][0]['value']]
        full_after = [r for r in full_samples if 'Return ACTION:' in r['messages'][0]['value']]
        self.assertEqual(filt_after, full_after)
        filt_stops = [r for r in filtered_samples if r['messages'][1]['value'] == 'ACTION: stop']
        self.assertEqual(len(filt_stops), 2)
        self.assertLess(len(filtered_samples), len(full_samples))

    def test_soft_window_trapezoidal_prob_and_full_no_downsample(self):
        self.assertAlmostEqual(vln.soft_window_reason_prob(10, 15, 10), 0.5)
        self.assertAlmostEqual(vln.soft_window_reason_prob(10, 15, 11.25), 0.75)
        self.assertAlmostEqual(vln.soft_window_reason_prob(10, 15, 15), 0.3)

        self.frames('scene_1', 20)
        ep = self.episode(actions=[-1] + [1, 2, 3] * 6 + [1, 0])
        index = {'["r2r","1",0]': {'instruction': 'go', 'segments': [
            {'type': 'only_act', 'start_step': 0, 'end_step': 8, 'content': None, 'updated_content': None},
            {'type': 'newroom', 'start_step': 8, 'end_step': 13, 'index': 8,
             'content': None, 'updated_content': 'Entered hallway. The next plan is to move forward.'},
            {'type': 'only_act', 'start_step': 13, 'end_step': 20,
             'content': 'Entered hallway. The next plan is to move forward.', 'updated_content': None},
        ]}}
        samples = []
        stats = vln.Counter()
        with patch.object(vln, 'MAX_ACTIONS_PER_SAMPLE', 1), patch.object(vln, 'COMBINE_PROB', 0.0):
            vln.convert_episode_vln(
                ep, self.root, samples, cot_index=index, dataset='r2r',
                soft_window=True, soft_window_pre_frames=1,
                direct_action_keep_prob=1.0, stop_repeat=2,
                empty_memory_drop_prob=0.0, sampling_rng=random.Random(7), stats=stats,
            )
        # All 5 frames in [7, 12) (i.e. index-1 .. index+3) must be represented and no only_act dropped.
        frame_names = {Path(r['images'][-1]).name for r in samples}
        for step in range(20):
            self.assertIn(f'{step:03d}.jpg', frame_names)
        self.assertEqual(stats['dropped_direct_move'], 0)
        self.assertGreater(stats['choice_reason'], 0)
        self.assertEqual(stats['choice_reason'], stats['action_after_reason'])
        # Stop at final frame is preserved and repeated once (total 2 exact stop copies).
        stops = [r for r in samples if r['messages'][1]['value'] == 'ACTION: stop']
        self.assertEqual(len(stops), 2)
        # Post-window frames carry updated memory with steps_ago > 0 and stripped stale plan.
        late = [r for r in samples if Path(r['images'][-1]).name == '016.jpg']
        self.assertEqual(len(late), 1)
        self.assertIn('Entered hallway.', late[0]['messages'][0]['value'])
        self.assertNotIn('The next plan is to move forward.', late[0]['messages'][0]['value'])

    def test_soft_window_cli_end_to_end(self):
        self.frames('scene_1', 11)
        ep = self.episode(actions=[-1, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 0])
        ann = self.root / 'r2r.json'
        ann.write_text(json.dumps([ep]))
        cot = self.root / 'cot_index.json'
        cot.write_text(json.dumps({'["r2r","1",0]': {'instruction': 'go', 'segments': [
            {'type': 'newroom', 'start_step': 4, 'end_step': 9, 'index': 4,
             'content': None, 'updated_content': 'Entered kitchen.'},
        ]}}))
        out = self.root / 'out.json'
        result = subprocess.run(
            [sys.executable, str(ROOT / 'tool/convert_vln.py'),
             '--annotation_path', str(ann), '--image_root', str(self.root),
             '--output_path', str(out), '--cot-index', str(cot), '--dataset', 'r2r',
             '--soft-window', '--soft-window-pre-frames', '1',
             '--direct-action-keep-prob', '1.0', '--stop-repeat', '2',
             '--empty-memory-drop-prob', '0.1'],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = json.loads(out.read_text())
        self.assertTrue(any(r['messages'][1]['value'].startswith('REASON: ') for r in rows))
        self.assertTrue(any(r['messages'][1]['value'] == 'ACTION: stop' for r in rows))

    def test_soft_window_post_entry_plan_alignment_and_nonzero_choice_age(self):
        # Actions after sentinel -1: 0=fwd, 1=fwd, 2=left, 3=fwd, 4=right, 5=fwd, 6=fwd, 7=stop
        self.frames('scene_1', 8)
        ep = self.episode(actions=[-1, 1, 1, 2, 1, 3, 1, 1, 0])
        raw_node = (
            'I see a hallway. The agent has exited the bedroom. '
            'The next step is to turn left to proceed down the hall. '
            'The next plan is to turn left.'
        )
        index = {'["r2r","1",0]': {'instruction': 'go', 'segments': [
            {'type': 'only_act', 'start_step': 0, 'end_step': 2, 'content': None, 'updated_content': None},
            {'type': 'newroom', 'start_step': 2, 'end_step': 7, 'index': 2,
             'content': None, 'updated_content': raw_node},
            {'type': 'only_act', 'start_step': 7, 'end_step': -1, 'content': raw_node, 'updated_content': None},
        ]}}
        expected_plan_by_frame = {
            '002.jpg': 'The next plan is to turn left.',
            '003.jpg': 'The next plan is to move forward.',
            '004.jpg': 'The next plan is to turn right.',
            '005.jpg': 'The next plan is to move forward.',
            '006.jpg': 'The next plan is to move forward.',
        }
        samples = []
        for seed in range(20):
            with patch.object(vln, 'MAX_ACTIONS_PER_SAMPLE', 1), patch.object(vln, 'COMBINE_PROB', 0.0):
                vln.convert_episode_vln(
                    ep, self.root, samples, cot_index=index, dataset='r2r',
                    soft_window=True, sampling_rng=random.Random(seed),
                )
        for i, row in enumerate(samples):
            frame = Path(row['images'][-1]).name
            user_text = row['messages'][0]['value']
            reply = row['messages'][1]['value']
            # Redundant prescriptive sentence in progress_assessment must never survive.
            self.assertNotIn('The next step is to turn left', user_text)
            self.assertNotIn('The next step is to turn left', reply)
            # Pre-entry frame 001.jpg must never emit REASON under default pre_frames=0 ([2, 7)).
            if frame == '001.jpg':
                self.assertFalse(reply.startswith('REASON: '))
            # Choice prompts must never carry "updated 0 steps ago".
            if 'Choose whether' in user_text:
                self.assertNotIn('updated 0 steps ago', user_text)
                self.assertNotIn('The next plan is to ', user_text)
            # Every REASON turn and its paired action_only (steps_ago=0) turn must match frame's expert action.
            if reply.startswith('REASON: '):
                self.assertIn(frame, expected_plan_by_frame)
                expected_plan = expected_plan_by_frame[frame]
                self.assertTrue(reply.endswith(expected_plan), (frame, reply))
                follow_up = samples[i + 1]
                self.assertIn('Return ACTION:', follow_up['messages'][0]['value'])
                self.assertIn('updated 0 steps ago', follow_up['messages'][0]['value'])
                self.assertIn(expected_plan, follow_up['messages'][0]['value'])

    def test_missing_frames_are_not_padded_for_movement(self):
        self.frames('scene_1', 1)
        with self.assertWarns(UserWarning):
            self.assertIsNone(common.prep_actions_and_images(self.episode(), self.root))
        output = []
        with self.assertWarns(UserWarning):
            vln.convert_episode_vln(self.episode(), self.root, output)
        self.assertEqual(output, [])

    def test_action_sequences_reject_strings_objects_and_scalars(self):
        self.frames('scene_1', 4)
        for field in ('actions', 'executed_actions'):
            for value in ('123', {'1': 1, '2': 2, '3': 3}, 123, True):
                with self.subTest(field=field, value=value), self.assertWarnsRegex(UserWarning, 'nonempty list'):
                    output = []
                    vln.convert_episode_vln(self.episode(**{field: value}), self.root, output)
                    self.assertEqual(output, [])

    def test_frame_gaps_and_duplicate_indices_are_rejected(self):
        folder = self.frames('scene_1', 4)
        (folder / '001.jpg').unlink()
        with self.assertWarns(UserWarning):
            self.assertIsNone(common.prep_actions_and_images(self.episode(), self.root))
        (folder / '001.jpg').touch(); (folder / '001.png').touch()
        with self.assertRaises(ValueError): common.list_episode_images(self.root, 'scene_1')

    def test_only_terminal_stop_can_duplicate_a_frame(self):
        self.frames('scene_1', 4)
        actions, images = common.prep_actions_and_images(self.episode(actions=[-1, 1, 2, 3, 0, 0]), self.root)
        self.assertEqual(actions[-1], 'stop')
        self.assertEqual(images[-2], images[-1])
        self.assertEqual(len(images), 5)
        self.assertIsNone(common.try_combine('stop', 'stop'))
        with self.assertWarns(UserWarning):
            self.assertIsNone(common.prep_actions_and_images(self.episode(actions=[-1, 1, 0, 2]), self.root))

    def test_numeric_string_actions_and_history_one(self):
        self.assertEqual(common.to_action_text('1'), 'forward 25 cm')
        self.assertEqual(common.to_action_text('0'), 'stop')
        self.assertEqual(common.uniform_sample_with_ends(list(range(4)), 1), [3])
        with self.assertRaises(ValueError): common.uniform_sample_with_ends([1], 0)

    def test_atomic_labels_use_execution_and_final_stop_frame(self):
        self.frames('scene_1', 4)
        ep = self.episode(actions=[-1, 1, 1, 1], executed_actions=[2, 1, 3, 0])
        output = []
        dagger.convert_episode_atomic(ep, self.root, output)
        self.assertTrue(output)
        self.assertIn('turn left 15 degree', output[0]['messages'][1]['value'])
        self.assertIn('stop', output[-1]['messages'][1]['value'])
        self.assertTrue(output[-1]['images'][-1].endswith('003.jpg'))

    def test_cli_creates_one_atomic_output_without_model_groups(self):
        self.frames('scene_1', 4)
        annotations = self.root / 'input.json'
        annotations.write_text('\ufeff\n ' + json.dumps([self.episode()]))
        output = self.root / 'new' / 'output.json'
        result = subprocess.run([sys.executable, str(ROOT / 'tool/convert_dagger.py'),
                                 '--annotation_path', str(annotations), '--image_root', str(self.root),
                                 '--output_path', str(output)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(output.read_text()))
        self.assertEqual(list(output.parent.iterdir()), [output])

    def test_standard_samples_match_original_converter(self):
        fixture = json.loads((ROOT / 'tests/fixtures/conversion_reference.json').read_text())
        episodes = fixture['episodes']
        for episode in episodes:
            self.frames(episode['video_id'], fixture['frame_count'])
        random.seed(fixture['seed']); result = []
        for ep in episodes:
            vln.convert_episode_vln(ep, self.root, result)
        common.relativize_image_paths(result, self.root)
        self.assertEqual(fixture['expected']['vln'], result)
        for sample in result:
            self.assertLessEqual(len(sample['images']), 9)
            self.assertGreaterEqual(len(sample['images']), 2)
            self.assertNotIn('image_roles', sample)
            self.assertNotIn('local observation', sample['messages'][0]['value'])
            self.assertEqual(sample['messages'][0]['value'].count('<image>'), len(sample['images']))

    def test_history_plus_current_indices(self):
        self.assertEqual(vln.sample_history_plus_current(list(range(12))),
                         [0, 1, 3, 4, 6, 7, 9, 10, 11])
        self.assertEqual(vln.sample_history_plus_current([0]), [0])

    def test_instruction_objects_are_read_as_text_and_bad_types_are_skipped(self):
        self.frames('scene_1', 4)
        output = []
        vln.convert_episode_vln(self.episode(instruction={'instruction_text': 'walk to the door'}), self.root, output)
        self.assertTrue(output)
        self.assertIn('walk to the door', output[0]['messages'][0]['value'])
        with self.assertWarns(UserWarning):
            self.assertEqual(common.get_instructions(self.episode(instruction=42)), [])

    def test_removed_modes_and_task_selection_are_rejected(self):
        output = self.root / 'output.json'
        output.write_text('previous output')
        cases = [('convert_dagger.py', '--mode', mode) for mode in ('merged', 'both')]
        cases += [('convert_vln.py', '--task_type', task) for task in ('idm', 'ffs')]
        for converter, flag, value in cases:
            with self.subTest(converter=converter, value=value):
                result = subprocess.run([sys.executable, str(ROOT / 'tool' / converter),
                    '--annotation_path', str(self.root / 'input.json'), '--image_root', str(self.root),
                    '--output_path', str(output), flag, value], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(value, result.stderr)
                self.assertEqual(output.read_text(), 'previous output')

    def test_cli_can_export_portable_image_paths(self):
        self.frames('scene_1', 4)
        annotations = self.root / 'input.json'
        annotations.write_text(json.dumps([self.episode(model_call_groups=[dict(call_step_id=0, actions=[1,2,3])])]))
        for converter in ('convert_vln.py', 'convert_dagger.py'):
            with self.subTest(converter=converter):
                output = self.root / converter / 'out.json'
                result = subprocess.run([sys.executable, str(ROOT / 'tool' / converter),
                    '--annotation_path', str(annotations), '--image_root', str(self.root),
                    '--output_path', str(output), '--relative_to', str(self.root)],
                    capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                for path in output.parent.glob('*.json'):
                    samples = json.loads(path.read_text())
                    self.assertTrue(samples)
                    for sample in samples:
                        for image in sample['images']:
                            self.assertFalse(Path(image).is_absolute())
                            self.assertTrue((self.root / image).is_file())

    def test_relative_export_rejects_images_outside_root(self):
        samples = [{'images': [str(self.root.parent / 'outside.jpg')]}]
        with self.assertRaisesRegex(ValueError, 'outside --relative_to'):
            common.relativize_image_paths(samples, self.root)

    def test_failed_serialization_keeps_existing_output(self):
        output = self.root / 'output.json'
        output.write_text('previous output')
        with self.assertRaises(TypeError):
            common.write_samples(output, [{'not_json': object()}])
        self.assertEqual(output.read_text(), 'previous output')
        self.assertEqual(list(self.root.iterdir()), [output])


if __name__ == '__main__': unittest.main()

"""CPU-only tests: replay resume, alignment rejection, and namespace-safe export."""
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from prepare_full import prepare, save
from run_full import export


class FullPipelineTests(unittest.TestCase):
    def test_replay_resume_and_failed_alignment_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'annotations/simplevln/sft/r2r/annotations.json'
            episodes = [dict(id=i, video=f'scene/{i}', actions=[-1, 2, 1], instructions=['Go.']) for i in (1, 2)]
            save(source, episodes)
            original = root/'general_vln_dataset/r2r/train/train.json.gz'
            original.parent.mkdir(parents=True)
            with gzip.open(original, 'wt') as stream:
                json.dump(dict(episodes=[dict(episode_id=i, scene_id='scene/scene.glb', start_position=[0, 0, 0]) for i in (1, 2)]), stream)
            scene = root/'scene/scene/scene.glb'
            scene.parent.mkdir(parents=True)
            scene.write_bytes(b'fake scene')
            for i in (1, 2):
                folder = root/f'images/r2r/scene/{i}/rgb'
                folder.mkdir(parents=True)
                for t in range(3):
                    (folder/f'{t}.jpg').write_bytes(b'fake RGB for mocked renderer')
            args = SimpleNamespace(dataset='r2r', data_root=str(root), output=str(root/'out'), retry_rejected=False)
            calls = []
            class FakeReplay:
                def replay(self, scene, original, actions, images, resolution, yaw=None):
                    calls.append((original['episode_id'], resolution))
                    assert actions == ['turn left 15 degree', 'forward 25 cm', 'stop']
                    assert len(images) == 3
                    passed = original['episode_id'] == 1
                    return dict(alignment_status='numeric_pass' if passed else 'rejected', yaw=0,
                                events=[], mean_mae=0 if passed else 30, max_mae=0 if passed else 50)
                def close(self):
                    pass
            with patch('prepare_full.Replayer', FakeReplay):
                prepare(args)
            self.assertEqual(calls, [(1, (120, 160)), (2, (120, 160)), (2, (480, 640))])
            ready = root/'out/annotations.ready.json'
            self.assertEqual([a['id'] for a in json.loads(ready.read_text())], [1])
            stamp = ready.stat().st_mtime_ns
            with patch('prepare_full.Replayer', side_effect=AssertionError('Resume must not render')):
                prepare(args)
            self.assertEqual(ready.stat().st_mtime_ns, stamp)
            (root/'images/r2r/scene/1/rgb/0.jpg').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'inputs changed'):
                prepare(args)

    def test_export_keeps_same_id_from_both_datasets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for ds in ('r2r', 'rxr'):
                save(root/f'generation/{ds}/cot_index.json', {json.dumps([ds, '1', 0]): {'dataset': ds}})
            export(root)
            self.assertEqual(len(json.loads((root/'exports/cot_index.json').read_text())), 2)

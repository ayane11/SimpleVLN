#!/usr/bin/env python3
"""Replay source trajectories, gate RGB alignment, then select causal room events.

Imports Habitat only when uncached replay work is required. No teacher calls.
"""
import argparse
import fcntl
import gzip
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tool'))
from convert_common import get_episode_id, load_episodes, prep_actions_and_images, write_samples
from core import digest
from room_events import detect, RULE_VERSION


def identity(path):
    p = Path(path).resolve()
    s = p.stat()
    return [str(p), s.st_size, s.st_mtime_ns]


def save(path, value):
    # Preserve mtime: generation manifests must stay valid on unchanged resume.
    if path.exists() and json.loads(path.read_text()) == value:
        return
    write_samples(path, value)


class Replayer:
    def __init__(self):
        import habitat_sim
        import magnum
        import numpy
        from PIL import Image
        from scipy.optimize import minimize_scalar
        from habitat_sim.utils.common import quat_from_angle_axis
        self.h, self.mn, self.np, self.Image = habitat_sim, magnum, numpy, Image
        self.minimize, self.quat = minimize_scalar, quat_from_angle_axis
        self.sim, self.scene, self.resolution = None, None, None

    def close(self):
        if self.sim:
            self.sim.close()
            self.sim = None

    def open(self, scene, resolution):
        if self.sim and (self.scene, self.resolution) == (scene, resolution):
            return
        self.close()
        h = self.h
        cfg = h.SimulatorConfiguration()
        cfg.scene_id, cfg.gpu_device_id, cfg.enable_physics = scene, 0, False
        sensor = h.CameraSensorSpec()
        sensor.uuid, sensor.resolution = 'rgb', list(resolution)
        sensor.position, sensor.hfov = self.mn.Vector3(0, 1.25, 0), 79
        ac = h.agent.AgentConfiguration()
        ac.sensor_specifications = [sensor]
        ac.action_space = {k: h.agent.ActionSpec(k, h.agent.ActuationSpec(amount=v))
                           for k, v in [('move_forward', .25), ('turn_left', 15), ('turn_right', 15)]}
        self.sim = h.Simulator(h.Configuration(cfg, [ac]))
        self.scene, self.resolution = scene, resolution

    def replay(self, scene, original, actions, images, resolution, yaw=None):
        np, Image = self.np, self.Image
        self.open(scene, resolution)
        agent = self.sim.initialize_agent(0)
        state = self.h.AgentState()
        state.position = original['start_position']
        def saved(path):
            with Image.open(path) as im:
                return np.array(im.convert('RGB').resize((160, 120), Image.Resampling.BOX))
        first = saved(images[0])
        def rendered():
            rgb = self.sim.get_sensor_observations()['rgb'][:, :, :3]
            return np.array(Image.fromarray(rgb).resize((160, 120), Image.Resampling.BOX))
        def set_yaw(degrees):
            state.rotation = self.quat(np.deg2rad(degrees), np.array([0, 1, 0]))
            agent.set_state(state)
        def loss(degrees):
            set_yaw(degrees)
            return float(np.abs(rendered().astype(float) - first).mean())
        if yaw is None:
            coarse = min(range(0, 360, 5), key=loss)
            fit = self.minimize(loss, bounds=(coarse-5, coarse+5), method='bounded', options={'xatol': .005})
            yaw = float(fit.x)
        set_yaw(yaw)
        regions = [r for r in self.sim.semantic_scene.regions if r is not None]
        steps = []
        names = {'forward 25 cm': 'move_forward', 'turn left 15 degree': 'turn_left', 'turn right 15 degree': 'turn_right'}
        traveled, last = 0., None
        for i, action in enumerate(actions):
            pos = agent.get_state().position.copy()
            if last is not None:
                traveled += float(np.linalg.norm(pos-last))
            last = pos
            candidates = [dict(id=r.id, category=r.category.name()) for r in regions
                          if r.aabb.contains(self.mn.Vector3(*(pos+np.array([0, .5, 0]))))]
            steps.append(dict(step=i, position=pos.tolist(), region_candidates=candidates,
                              traveled_m=traveled, rgb_mae=float(np.abs(rendered().astype(float)-saved(images[i])).mean())))
            # Image i is PRE-action i, using the same normalized indexing as SFT.
            if action != 'stop':
                self.sim.step(names[action])
        for row in steps:
            row['progress_ratio'] = row['traveled_m']/traveled if traveled else 0.
        maes = [row['rgb_mae'] for row in steps]
        mean, maximum = float(np.mean(maes)), max(maes)
        passed = mean < 10 and maximum < 20
        return dict(yaw=yaw, steps=steps, mean_mae=mean, max_mae=maximum,
                    render_resolution=list(resolution), alignment_status='numeric_pass' if passed else 'rejected',
                    events=detect(steps) if passed else [])


def prepare(args):
    root, out = Path(args.data_root), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    ds = args.dataset
    annotations = root / f'annotations/simplevln/sft/{ds}/annotations.json'
    original_path = root / f'general_vln_dataset/{ds}/train' / ('train.json.gz' if ds == 'r2r' else 'train_guide.json.gz')
    episodes = load_episodes(annotations)
    with gzip.open(original_path, 'rt') as stream:
        originals = {str(e['episode_id']): e for e in json.load(stream)['episodes']}
    ids = [get_episode_id(a) for a in episodes]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate episode IDs')
    config = dict(annotations=identity(annotations), originals=identity(original_path),
                  code=digest([Path(__file__).read_text(), (HERE/'room_events.py').read_text(),
                               (HERE.parent/'tool/convert_common.py').read_text()]), rule=RULE_VERSION)
    manifest = out/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError('Preparation inputs/code changed: use a new output root')
    save(manifest, config)
    ready, events, excluded = [], {}, []
    engine = None
    # Group by scene to avoid repeatedly loading meshes.
    episodes.sort(key=lambda a: (originals.get(get_episode_id(a), {}).get('scene_id', ''), get_episode_id(a)))
    try:
        for a in episodes:
            eid = get_episode_id(a)
            target = out/'replays'/f'{digest(eid)}.json'
            try:
                original = originals[eid]
                scene = root/'scene'/original['scene_id']
                prepared = prep_actions_and_images(a, str(root/'images'/ds))
                if prepared is None:
                    raise ValueError('Image/action alignment failed')
                actions, images = prepared
                images = images[:len(actions)]
                if any(x not in {'stop', 'forward 25 cm', 'turn left 15 degree', 'turn right 15 degree'} for x in actions):
                    raise ValueError('Replay requires unmerged primitive expert actions')
                # Include semantic/navmesh assets as well as original RGB identities.
                signature = digest(dict(annotation=a, original=original, images=[identity(p) for p in images],
                                        scene_assets=[identity(p) for p in sorted(scene.parent.iterdir()) if p.is_file()]))
                record = json.loads(target.read_text()) if target.exists() else None
                if record and record.get('signature') != signature:
                    raise ValueError('Replay inputs changed; use a new output root')
                if not record or (args.retry_rejected and record['alignment_status'] == 'rejected'):
                    if engine is None:
                        engine = Replayer()
                    record = engine.replay(str(scene), original, actions, images, (120, 160))
                    if record['alignment_status'] == 'rejected':
                        record = engine.replay(str(scene), original, actions, images, (480, 640), record['yaw'])
                    record.update(dataset=ds, episode_id=eid, signature=signature,
                                  approval='automated RGB gate only; not human-reviewed')
                    save(target, record)
                if record['alignment_status'] == 'numeric_pass':
                    ready.append(a)
                    events[eid] = record['events']
                else:
                    excluded.append(dict(episode_id=eid, reason='RGB alignment gate', mean_mae=record['mean_mae'], max_mae=record['max_mae']))
                print(ds, eid, record['alignment_status'], flush=True)
            except (KeyError, ValueError, FileNotFoundError) as exc:
                # Changed cache identity is not a skippable dataset issue.
                if 'inputs changed' in str(exc):
                    raise
                excluded.append(dict(episode_id=eid, reason=str(exc)))
                print(ds, eid, 'EXCLUDED', str(exc), flush=True)
    finally:
        if engine:
            engine.close()
    save(out/'annotations.ready.json', ready)
    save(out/'events.json', events)
    save(out/'excluded.json', excluded)
    save(out/'status.json', dict(total=len(episodes), ready=len(ready), excluded=len(excluded), nodes=sum(map(len, events.values()))))
    if not ready:
        raise ValueError('No aligned trajectories; generation must not proceed')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['r2r', 'rxr'], required=True)
    p.add_argument('--data-root', default='/data/vln')
    p.add_argument('--output', required=True)
    p.add_argument('--retry-rejected', action='store_true')
    args = p.parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    with (Path(args.output)/'.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare(args)

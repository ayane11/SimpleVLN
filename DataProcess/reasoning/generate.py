#!/usr/bin/env python3
"""Generate auditable offline navigation reasoning; never starts SFT or simulation."""
import argparse
import fcntl
import json
import sys
import traceback
import time
import random
import threading
import copy
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version, PackageNotFoundError
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tool"))
from convert_common import (get_episode_id, get_video_id, get_instructions,
                            load_episodes, prep_actions_and_images, write_samples)
from core import (VERSION, SYSTEM, digest, sample_steps, parse_json, validate_nodes,
                  validate_reason, reason_text, make_segments, global_prompt, node_prompt, image_turn)
from teacher import TransformersTeacher, VLLMTeacher, check_weights
from core import validate_expert_reason, validate_memory_reason, memory_steps
from expert_progress import decomposition_prompt, validate_subtasks, progress_prompt, validate_progress
from response_reuse import load_responses, request_identity

REUSED_RESPONSES = {}


def file_identity(path):
    p = Path(path).resolve()
    stat = p.stat()
    return dict(path=str(p), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def atomic(path, value):
    write_samples(str(path), value)


def request_json(teacher, messages, validator, settings, output, retries):
    """Validated requests cached independently, so a failed node does not lose global work."""
    images = [file_identity(c['image']) for m in messages if isinstance(m['content'], list)
              for c in m['content'] if c['type'] == 'image']
    key = digest(dict(messages=messages, images=images, settings=settings))
    target = output / "requests" / f"{key}.json"
    if target.exists():
        cached = json.loads(target.read_text())
        validator(cached['parsed'])
        return cached['parsed'], key
    attempts = []
    original_messages = messages
    for attempt in range(retries + 1):
        response = None
        try:
            response = copy.deepcopy(REUSED_RESPONSES.get(request_identity(messages, images, settings)))
            if response is None:
                response = teacher.generate(messages)
            value = parse_json(response['text'])
            validator(value)
        except Exception as exc:
            attempts.append(dict(attempt=attempt, response=response, error=f"{type(exc).__name__}: {exc}"))
            atomic(output / "errors" / f"{key}.json", dict(messages=messages, settings=settings, attempts=attempts))
            if attempt == retries:
                raise
            # Make the repair explicit, without persisting an unbounded conversation.
            if settings.get('expert_progress_v3') and response and response.get('text'):
                messages = original_messages + [{"role": "assistant", "content": response['text']}]
            messages = messages + [{"role": "user", "content":
                f"The previous attempt failed validation: {exc}. Return the complete required JSON. "
                "Use only the supplied frame IDs and allowed node types."}]
        else:
            atomic(target, dict(request_id=key, messages=messages, settings=settings,
                                image_files=images, parsed=value, response=response,
                                preceding_failed_attempts=attempts))
            return value, key


def plan_episode(ep, instruction_index, image_root, args, events):
    prepared = prep_actions_and_images(ep, image_root)
    if prepared is None:
        raise ValueError("Frame/action alignment failed; see preceding converter warning")
    actions, images = prepared
    # Observations before executable actions; discard only the optional post-STOP duplicate.
    images = [str(Path(p).resolve()) for p in images[:len(actions)]]
    instructions = get_instructions(ep)
    instruction = instructions[instruction_index]
    episode_id = get_episode_id(ep)
    key = json.dumps([args.dataset, episode_id, instruction_index], separators=(',', ':'))
    supplied = None
    if events is not None:
        # Missing events is NOT permission to guess simulator labels.
        if episode_id not in events:
            raise ValueError(f"Missing events entry for episode {episode_id}; use [] for no nodes")
        supplied = validate_nodes(events[episode_id], len(images))
    global_steps = sample_steps(len(images)-1, args.global_frames)
    global_message = image_turn(global_prompt(instruction, len(images), args.max_nodes), images, global_steps)
    return dict(key=key, dataset=args.dataset, episode_id=episode_id, instruction_index=instruction_index,
                instruction=instruction, video=get_video_id(ep), step_convention="zero-based pre-action observation",
                source_annotation=ep,
                actions=actions, images=images, image_files=[file_identity(p) for p in images],
                global_steps=global_steps, global_message=global_message, supplied_nodes=supplied,
                node_source="provided_events" if supplied is not None else "teacher_visual_proposal")


def annotate(plan, teacher, args, settings, output):
    frames, images = len(plan['images']), plan['images']
    causal = plan['supplied_nodes'] is not None
    base = [{"role": "system", "content": SYSTEM}]
    if not causal:
        base.append(plan['global_message'])

    def check_global(v):
        if not isinstance(v.get('episode_summary'), str) or not v['episode_summary'].strip():
            raise ValueError("Missing episode_summary")
        validate_nodes(v.get('nodes'), frames, proposed=True, visible_steps=plan['global_steps'], max_nodes=args.max_nodes)

    overview, global_id = ({'episode_summary': '', 'nodes': []}, None) if causal else request_json(teacher, base, check_global, settings, output, args.retries)
    nodes = plan['supplied_nodes']
    if nodes is None:
        nodes = validate_nodes(overview['nodes'], frames, proposed=True,
                               visible_steps=plan['global_steps'], max_nodes=args.max_nodes)
    # Retain the global visual turn (privileged). Bound each node conversation instead of
    # accumulating every earlier node's images. Pass latest reasoning explicitly below.
    if not causal:
        base.append({"role": "assistant", "content": json.dumps(overview, ensure_ascii=False)})
    results, previous, previous_step = [], None, None
    structured = causal and getattr(args, 'expert_progress_v3', False)
    subtasks, subtasks_id, previous_state = [], None, None
    if structured and nodes:
        decomposition, subtasks_id = request_json(teacher, base + [{"role": "user", "content": decomposition_prompt(plan['instruction'])}],
            lambda v: validate_subtasks(v, plan['instruction']), settings, output, args.retries)
        subtasks = decomposition['subtasks']
    failed_nodes, excluded_nodes = [], []
    exclusions = getattr(args, 'node_exclusions', {}).get(plan.get('dataset'), {}).get(plan.get('episode_id'), {})
    for node in nodes:
        if str(node['step']) in exclusions:
            excluded_nodes.append(dict(step=node['step'], reason=exclusions[str(node['step'])]))
            continue
        memory_update = causal and (getattr(args, 'memory_update_v2', False) or structured)
        history = (memory_steps(node['step'], args.history_frames, previous_step) if memory_update
                   else sample_steps(node['step'], args.history_frames))
        prompt = node_prompt(plan['instruction'], node, previous, previous_step, plan['actions'], history,
                             memory_update=memory_update)
        if structured:
            prompt = progress_prompt(plan['instruction'], node, plan['actions'], history, subtasks, previous_state, previous_step)
        current = image_turn(prompt, images, history, 'causal history/current')
        end = node.get('correction_end_step')
        future = []
        if end is not None and end > node['step']:
            relative = sample_steps(end-node['step']-1, args.correction_frames)
            future = [i + node['step'] + 1 for i in relative]
            current['content'].extend(image_turn(
                "Privileged future correction observations; NEVER cite as current evidence.",
                images, future, 'future correction')['content'])
        try:
            if structured:
                validator = lambda v: validate_progress(v, history, plan['actions'][node['step']], subtasks, previous_state)
            elif memory_update:
                validator = lambda v: validate_memory_reason(v, history, plan['actions'][node['step']:node['step']+3], previous)
            elif causal:
                validator = lambda v: validate_expert_reason(v, history, plan['actions'][node['step']:node['step']+3])
            else:
                validator = lambda v: validate_reason(v, history)
            value, request_id = request_json(teacher, base+[current],
                                             validator, settings, output, args.retries)
        except Exception as exc:
            if not causal:
                raise
            failed_nodes.append(dict(step=node['step'], error=str(exc)))
            continue
        results.append(dict(**node, reasoning=value, history_steps=history, correction_steps=future,
                            previous_reasoning=previous, previous_reasoning_step=previous_step,
                            previous_progress_state=previous_state,
                            request_id=request_id, review_status="pending"))
        if structured:
            previous_state = value['progress_state']
        previous, previous_step = reason_text(value), node['step']
    return dict(**{k: v for k, v in plan.items() if k != 'global_message'},
                schema_version=VERSION, generation_status="complete", review_status="pending",
                teacher=settings, generated_at=datetime.now(timezone.utc).isoformat(),
                global_request_id=global_id, episode_summary=overview['episode_summary'], nodes=results,
                segments=make_segments(results, frames),
                teacher_has_future_context=not causal, failed_nodes=failed_nodes, excluded_nodes=excluded_nodes,
                instruction_subtasks=subtasks, subtasks_request_id=subtasks_id,
                warning="Draft labels. Schema/evidence-ID checks do not establish semantic correctness.")


def export_cot(output):
    cot = {}
    for path in sorted((output / 'episodes').glob('*.json')):
        record = json.loads(path.read_text())
        if record.get('generation_status') != 'complete':
            continue
        if record['key'] in cot:
            raise ValueError("Duplicate episode key in output")
        cot[record['key']] = {k: record[k] for k in
                             ('episode_id', 'dataset', 'instruction_index', 'instruction', 'segments', 'review_status')}
    atomic(output / 'cot_index.json', cot)
    native = {}
    for record in cot.values():
        eid = str(record['episode_id'])
        if eid in native:
            raise ValueError('Aware export cannot represent multiple instructions under the same episode ID')
        native[eid] = {'segments': record['segments']}
    atomic(output / 'cot_new.json', native)
    return len(cot)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations', required=True)
    p.add_argument('--image-root', required=True)
    p.add_argument('--dataset', required=True, help='Dataset namespace, e.g. r2r, rxr, scalevln')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--model', default=str(HERE.parents[2] / 'models' / 'Qwen3.5-27B'),
                   help='Local downloaded model; use explicit /workspace/VLN/models/Qwen3.5-27B in Docker')
    p.add_argument('--backend', choices=['transformers', 'vllm'], default='transformers')
    p.add_argument('--base-url', default='http://127.0.0.1:22990/v1')
    p.add_argument('--served-model', default='qwen35-27b')
    p.add_argument('--base-urls', nargs='+', help='Local vLLM replicas; workers assigned round robin')
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--sample-size', type=int, default=0, help='Seeded random episode subset; resume without this flag to extend')
    p.add_argument('--seed', type=int, default=20261117)
    p.add_argument('--events-json', help='JSON object episode ID -> node list, zero-based steps')
    p.add_argument('--episode-id', action='append', help='Repeat to select episodes')
    p.add_argument('--limit', type=int, default=0, help='Maximum trajectory/instruction pairs, 0=all')
    p.add_argument('--global-frames', type=int, default=24, help='0=all (paper full-episode input); default downsamples')
    p.add_argument('--history-frames', type=int, default=8)
    p.add_argument('--memory-update-v2', action='store_true', help='Causal room nodes: recent consecutive frames, explicit memory update and stricter validation')
    p.add_argument('--expert-progress-v3', action='store_true', help='Same nodes, strict current action/STOP and evidenced subtask memory')
    p.add_argument('--retry-failed', action='store_true', help='Retry incomplete episodes using successful request caches')
    p.add_argument('--exclude-nodes', help='JSON dataset -> episode ID -> step -> reason')
    p.add_argument('--reuse-requests-from', help='Reuse exact model requests with matching image identities and generation settings; revalidate every response')
    p.add_argument('--correction-frames', type=int, default=8)
    p.add_argument('--max-nodes', type=int, default=0, help='0=no arbitrary node-count cap')
    p.add_argument('--max-pixels', type=int, default=262144)
    p.add_argument('--max-new-tokens', type=int, default=1536)
    p.add_argument('--max-context', type=int, default=32768)
    p.add_argument('--gpu-memory-gib', type=int, default=40, help='Weight placement budget per visible GPU, leaves runtime headroom')
    p.add_argument('--timeout', type=int, default=300)
    p.add_argument('--retries', type=int, default=2)
    p.add_argument('--dry-run', action='store_true', help='Validate alignment and save request plans; no GPU/model calls')
    args = p.parse_args()
    if min(args.limit, args.global_frames, args.retries, args.max_nodes, args.sample_size) < 0 or min(
        args.workers, args.history_frames, args.correction_frames, args.max_new_tokens, args.timeout, args.gpu_memory_gib
    ) < 1 or args.max_pixels < 65536 or args.max_context <= args.max_new_tokens:
        p.error('Invalid limits/frame counts/token budget')
    if args.backend != 'vllm' and (args.workers != 1 or args.base_urls):
        p.error('Concurrent workers/replicas require vllm')
    if args.expert_progress_v3 and not args.events_json:
        p.error('--expert-progress-v3 requires supplied causal events')
    output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    # One writer per output directory, preventing duplicate expensive teacher calls.
    with (output / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args, output)


def run(args, output):
    global REUSED_RESPONSES
    args.node_exclusions = json.loads(Path(args.exclude_nodes).read_text()) if getattr(args, 'exclude_nodes', None) else {}
    events = json.loads(Path(args.events_json).read_text()) if args.events_json else None
    if events is not None and not isinstance(events, dict):
        raise ValueError('events-json must be an object mapping episode IDs to node lists')
    settings = {k: getattr(args, k) for k in ('model', 'backend', 'base_url', 'served_model', 'max_pixels',
                'max_new_tokens', 'max_context', 'global_frames', 'history_frames', 'correction_frames', 'max_nodes',
                'gpu_memory_gib')}
    settings.update(expert_progress_v3=getattr(args, 'expert_progress_v3', False), memory_update_v2=getattr(args, 'memory_update_v2', False), schema_version=VERSION, thinking=False, decoding='greedy',
                    dtype='bfloat16',
                    code_hash=digest({n: (HERE/n).read_text() for n in ('core.py', 'teacher.py', 'generate.py', 'expert_progress.py', 'response_reuse.py')}))
    settings['library_versions'] = {}
    for package in ('torch', 'transformers', 'accelerate', 'vllm'):
        try:
            settings['library_versions'][package] = version(package)
        except PackageNotFoundError:
            settings['library_versions'][package] = None
    model = Path(args.model)
    if model.exists():
        settings['model_files'] = [file_identity(p) for p in sorted(model.glob('*.json'))]
        settings['weight_files'] = [file_identity(p) for p in sorted(model.glob('*.safetensors'))]
    reuse_from = getattr(args, 'reuse_requests_from', None)
    settings['reuse_requests_from'] = str(Path(reuse_from).resolve()) if reuse_from else None
    REUSED_RESPONSES = load_responses(reuse_from, settings) if reuse_from else {}
    config = dict(settings=settings, annotations=file_identity(args.annotations),
                  image_root=str(Path(args.image_root).resolve()), dataset=args.dataset, events=events, exclusions=args.node_exclusions)
    manifest = output / 'manifest.json'
    if not args.dry_run:
        if args.backend == 'transformers':
            check_weights(args.model)
        if manifest.exists() and json.loads(manifest.read_text()) != config:
            raise ValueError('Output belongs to different data/model/settings/code; use a new output-dir')
        if not manifest.exists():
            atomic(manifest, config)
    started = time.monotonic()
    episodes = load_episodes(args.annotations)
    if args.episode_id:
        episodes = [ep for ep in episodes if get_episode_id(ep) in args.episode_id]
    if args.sample_size:
        episodes = random.Random(args.seed).sample(episodes, min(args.sample_size, len(episodes)))
    tasks, seen_keys = [], set()
    for ep in episodes:
        if not get_instructions(ep):
            raise ValueError(f'No instruction: {get_episode_id(ep)}')
        for instruction_index in range(len(get_instructions(ep))):
            key = json.dumps([args.dataset, get_episode_id(ep), instruction_index], separators=(',', ':'))
            if key in seen_keys:
                raise ValueError(f'Duplicate trajectory/instruction key: {key}')
            seen_keys.add(key)
            tasks.append((ep, instruction_index, key))
    if args.limit:
        tasks = tasks[:args.limit]
    if not tasks:
        raise ValueError('No trajectory/instruction pairs selected')
    atomic(output / 'selection.json', dict(seed=args.seed, keys=[t[2] for t in tasks]))
    local = threading.local()
    assignment_lock = threading.Lock()
    assignment = [0]
    endpoints = args.base_urls or [args.base_url]

    def process(task):
        ep, instruction_index, key = task
        t0 = time.monotonic()
        target = output / 'episodes' / f'{digest(key)}.json'
        try:
            plan = plan_episode(ep, instruction_index, args.image_root, args, events)
            if args.dry_run:
                atomic(output / 'plans' / f'{digest(key)}.json', plan)
                return dict(key=key, status='dry_run', seconds=time.monotonic()-t0)
            if target.exists():
                old = json.loads(target.read_text())
                if old['image_files'] != plan['image_files'] or old['instruction'] != plan['instruction']:
                    raise ValueError('Episode inputs changed; use a new output-dir')
                if not (getattr(args, 'retry_failed', False) and old.get('failed_nodes')):
                    print(f'RESUME {key}', flush=True)
                    return dict(key=key, status='cached', seconds=time.monotonic()-t0, failed_nodes=len(old.get('failed_nodes', [])))
            if not hasattr(local, 'teacher'):
                worker_args = copy.copy(args)
                with assignment_lock:
                    worker_args.base_url = endpoints[assignment[0] % len(endpoints)]
                    assignment[0] += 1
                local.teacher = TransformersTeacher(worker_args) if args.backend == 'transformers' else VLLMTeacher(worker_args)
            record = annotate(plan, local.teacher, args, settings, output)
            record['generation_seconds'] = time.monotonic()-t0
            atomic(target, record)
            print(f'OK {key}: {len(record["nodes"])} reasoning nodes, {record["generation_seconds"]:.1f}s', flush=True)
            return dict(key=key, status='generated', nodes=len(record['nodes']), failed_nodes=len(record['failed_nodes']), seconds=record['generation_seconds'])
        except Exception as exc:
            atomic(output / 'errors' / f'episode-{digest(key)}.json',
                   dict(key=key, error=str(exc), traceback=traceback.format_exc()))
            print(f'ERROR {key}: {exc}', file=sys.stderr, flush=True)
            return dict(key=key, status='failed', seconds=time.monotonic()-t0, error=str(exc))

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(process, tasks))
    failures = sum(r['status'] == 'failed' for r in results)
    summary = dict(wall_seconds=time.monotonic()-started, workers=args.workers, endpoints=endpoints,
                   processed=len(tasks), failures=failures, failed_nodes=sum(r.get('failed_nodes', 0) for r in results), results=results,
                   finished_at=datetime.now(timezone.utc).isoformat())
    atomic(output / 'run_summary.json', summary)
    if not args.dry_run:
        print(f'Exported {export_cot(output)} draft records to {output / "cot_new.json"}')
    print(f'Processed={len(tasks)}, failures={failures}; wall_seconds={summary["wall_seconds"]:.1f}')
    if failures or summary['failed_nodes']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

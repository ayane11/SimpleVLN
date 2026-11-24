"""Convert navigation demonstrations into visual instruction-following samples."""

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys

from tqdm import tqdm
from convert_common import (
    get_episode_id, try_combine, uniform_sample_with_ends, prep_actions_and_images,
    get_instructions, load_episodes, relativize_image_paths, write_samples,
)


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from navigation_memory import memory_prompt


SYSTEM_PROMPT = "You are a visual language navigation model."

VLN_PROMPT_TEMPLATE = (
    "Imagine you are a robot programmed for navigation tasks. "
    "You have been given a video of historical observations{} and an image of the current observation{}. "
    "Your assigned task is: '{}'. Analyze this series of images to decide your next move, "
    "which could involve turning left or right by a specific degree or moving forward a certain distance."
)

MAX_HISTORY = 8
MAX_ACTIONS_PER_SAMPLE = 3
COMBINE_PROB = 0.7


def normalize_pair(images):
    return [images[0], images[0]] if len(images) == 1 else images


def sample_history_plus_current(images, max_history=None):
    max_history = MAX_HISTORY if max_history is None else max_history
    if len(images) <= 1:
        return images
    history = images[:-1]
    if len(history) > max_history:
        history = uniform_sample_with_ends(history, max_history)
    return history + [images[-1]]


def build_vln_prompt(instruction, image_count):
    history_count = max(0, image_count - 1)
    return VLN_PROMPT_TEMPLATE.format("<image>" * history_count, "<image>", instruction)


def finalize_vln_sample(sample, instruction, max_history=None):
    user = sample["messages"][0]
    images = sample_history_plus_current(user["image"], max_history=max_history)
    images = normalize_pair(images)
    user["image"] = images
    user["value"] = build_vln_prompt(instruction, len(images))
    sample["images"] = list(images)
    del user["image"]
    return sample


def action_chunks(actions, boundaries):
    start = 0
    chunk = actions[0]
    for i, action in enumerate(actions[1:], 1):
        if i not in boundaries:
            combined = try_combine(chunk, action) if action == actions[i - 1] else None
            if combined is not None and random.random() <= COMBINE_PROB:
                chunk = combined
                continue
            if chunk.count(",") < MAX_ACTIONS_PER_SAMPLE - 1:
                chunk += ", " + action
                continue
        yield start, chunk
        start, chunk = i, action
    yield start, chunk


def first_action_chunk(actions, start, rng):
    """Build a single SimpleVLN action chunk starting at frame `start`."""
    chunk = actions[start]
    for i in range(start + 1, len(actions)):
        action = actions[i]
        combined = try_combine(chunk, action) if action == actions[i - 1] else None
        if combined is not None and rng.random() <= COMBINE_PROB:
            chunk = combined
            continue
        if chunk.count(",") < MAX_ACTIONS_PER_SAMPLE - 1:
            chunk += ", " + action
            continue
        break
    return chunk


def soft_window_reason_prob(start_step, end_step, now_step,
                            start_prob=0.5, peak_prob=0.75, end_prob=0.3):
    """AwareVLN-style trapezoidal reasoning probability over [start_step, end_step)."""
    step_range = end_step - start_step
    if step_range <= 0:
        return peak_prob
    peak_step = start_step + step_range / 4.0
    if now_step <= peak_step:
        ratio = (now_step - start_step) / (peak_step - start_step)
        return start_prob + (peak_prob - start_prob) * ratio
    ratio = (now_step - peak_step) / (end_step - peak_step)
    return peak_prob - (peak_prob - end_prob) * ratio


def _build_soft_windows(segments, total_steps, pre_frames=1, default_width=5):
    """Build non-overlapping soft-window descriptors for reasoning segments."""
    raw = []
    for segment in segments:
        if segment['type'] == 'only_act':
            continue
        step = segment['index']
        update = segment['updated_content']
        if not isinstance(update, str) or not update.strip():
            continue
        seg_start = segment.get('start_step', step)
        seg_end = segment.get('end_step', step + default_width)
        if seg_end <= seg_start:
            seg_end = seg_start + default_width
        w_start = max(0, seg_start - pre_frames)
        w_end = min(total_steps, max(w_start + 1, seg_end - pre_frames))
        if w_end <= step:
            w_end = min(total_steps, step + 1)
        raw.append({
            'index': step,
            'start_step': w_start,
            'end_step': w_end,
            'updated_content': update.strip(),
        })
    raw.sort(key=lambda item: item['index'])
    windows = []
    for item in raw:
        if windows and item['start_step'] < windows[-1]['end_step']:
            # Keep both windows non-overlapping; prioritize the later node's index.
            if windows[-1]['end_step'] > item['index']:
                windows[-1]['end_step'] = max(windows[-1]['start_step'] + 1, item['index'])
            item['start_step'] = max(item['start_step'], windows[-1]['end_step'])
            item['end_step'] = max(item['start_step'] + 1, item['end_step'])
            item['end_step'] = min(total_steps, item['end_step'])
        windows.append(item)
    return windows


def convert_episode_vln(episode, image_root, out_samples, max_history=None, *,
                        cot_index=None, dataset=None, action_control=False,
                        drop_pre_node_buffer=False, direct_action_keep_prob=1.0,
                        stop_repeat=1, soft_window=False, soft_window_pre_frames=1,
                        empty_memory_drop_prob=0.0, sampling_rng=None, stats=None):
    if not (0.0 < direct_action_keep_prob <= 1.0):
        raise ValueError(f"direct_action_keep_prob must be in (0, 1], got {direct_action_keep_prob!r}")
    if stop_repeat < 1:
        raise ValueError(f"stop_repeat must be >= 1, got {stop_repeat!r}")
    if soft_window_pre_frames < 0:
        raise ValueError(f"soft_window_pre_frames must be >= 0, got {soft_window_pre_frames!r}")
    if not (0.0 <= empty_memory_drop_prob < 1.0):
        raise ValueError(f"empty_memory_drop_prob must be in [0, 1), got {empty_memory_drop_prob!r}")

    prepared = prep_actions_and_images(episode, image_root)
    if prepared is None:
        return
    action_texts, images = prepared
    instructions = get_instructions(episode)
    if not instructions:
        return
    episode_id = get_episode_id(episode)
    rng = sampling_rng if sampling_rng is not None else random

    missing = 0
    for instruction_index, instruction in enumerate(instructions):
        nodes = {}
        soft_windows = []
        if cot_index is not None:
            key = json.dumps([dataset, episode_id, instruction_index], separators=(',', ':'))
            record = cot_index.get(key)
            if record is None:
                missing += 1
                continue
            if record['instruction'] != instruction:
                raise ValueError(f"Instruction mismatch for {key}")
            for segment in record['segments']:
                if segment['type'] == 'only_act':
                    continue
                step = segment['index']
                update = segment['updated_content']
                if (step in nodes or not 0 <= step < len(action_texts) or
                        not isinstance(update, str) or not update.strip()):
                    raise ValueError(f"Invalid reasoning node at {key}, step {step}")
                nodes[step] = update.strip()
            if soft_window:
                soft_windows = _build_soft_windows(
                    record['segments'], len(action_texts), pre_frames=soft_window_pre_frames
                )

        pre_node_steps = {step - 1 for step in nodes if step > 0}
        # Materialize chunks before sampling so RNG draws cannot alter chunk boundaries.
        chunks = list(action_chunks(action_texts, nodes))
        if cot_index is not None and not action_control and soft_window:
            # Build unified ordered events: every frame inside each 5-frame soft window
            # plus every only_act chunk outside all soft windows (no downsampling by default).
            window_frames = set()
            events = []
            for win in soft_windows:
                for t in range(win['start_step'], win['end_step']):
                    window_frames.add(t)
                    events.append((t, first_action_chunk(action_texts, t, rng), win))
            for start, action in chunks:
                if start not in window_frames:
                    events.append((start, action, None))
            events.sort(key=lambda item: item[0])

            memory = ''
            last_reason_step = None
            last_ref_range = None
            win_idx = 0
            for start, action, active_win in events:
                while win_idx < len(soft_windows) and start >= soft_windows[win_idx]['end_step']:
                    completed = soft_windows[win_idx]
                    memory = completed['updated_content']
                    last_reason_step = completed['index']
                    last_ref_range = (completed['start_step'], completed['end_step'])
                    win_idx += 1

                frames = normalize_pair(sample_history_plus_current(images[:start + 1], max_history))
                prompt = build_vln_prompt(instruction, len(frames))
                has_stop = 'stop' in action.lower()
                is_exact_stop = action.strip().lower() == 'stop'
                turns = []

                def _prev_steps_ago():
                    if not memory or last_ref_range is None:
                        return None
                    ref_lo, ref_hi = last_ref_range
                    upper = min(ref_hi, start + 1)
                    if ref_lo < upper:
                        ref_step = rng.randrange(ref_lo, upper)
                    else:
                        ref_step = last_reason_step if last_reason_step is not None else start
                    return max(0, start - ref_step)

                if active_win is not None:
                    w_start = active_win['start_step']
                    w_end = active_win['end_step']
                    new_memory = active_win['updated_content']
                    p_reason = soft_window_reason_prob(w_start, w_end, start)
                    if rng.random() < p_reason:
                        steps_ago = _prev_steps_ago()
                        decision = prompt + memory_prompt(memory, steps_ago=steps_ago)
                        turns.append((decision, 'REASON: ' + new_memory))
                        if stats is not None:
                            stats['choice_reason'] += 1

                        action_prompt = prompt + memory_prompt(new_memory, action_only=True, steps_ago=0)
                        action_reply = 'ACTION: ' + action
                        turns.append((action_prompt, action_reply))
                        if stats is not None:
                            stats['action_after_reason'] += 1
                            if has_stop:
                                stats['action_after_reason_stop'] += 1
                        if is_exact_stop and stop_repeat > 1:
                            for _ in range(stop_repeat - 1):
                                turns.append((action_prompt, action_reply))
                            if stats is not None:
                                stats['stop_repeat_extra'] += stop_repeat - 1
                    else:
                        # AwareVLN window action supervision: 30% old memory, 70% fresh new memory.
                        if rng.random() < 0.3:
                            ctx_memory = memory
                            ctx_steps_ago = _prev_steps_ago()
                        else:
                            ctx_memory = new_memory
                            ref_step = rng.randrange(w_start, start + 1)
                            ctx_steps_ago = max(0, start - ref_step)
                        decision = prompt + memory_prompt(ctx_memory, steps_ago=ctx_steps_ago)
                        action_reply = 'ACTION: ' + action
                        turns.append((decision, action_reply))
                        if stats is not None:
                            stats['choice_action_stop' if has_stop else 'choice_action_move'] += 1
                        if is_exact_stop and stop_repeat > 1:
                            for _ in range(stop_repeat - 1):
                                turns.append((decision, action_reply))
                            if stats is not None:
                                stats['stop_repeat_extra'] += stop_repeat - 1
                else:
                    if (not has_stop) and direct_action_keep_prob < 1.0 and rng.random() > direct_action_keep_prob:
                        if stats is not None:
                            stats['dropped_direct_move'] += 1
                        continue
                    if memory and empty_memory_drop_prob > 0.0 and rng.random() < empty_memory_drop_prob:
                        ctx_memory = ''
                        ctx_steps_ago = None
                        if stats is not None:
                            stats['empty_memory_dropped'] += 1
                    else:
                        ctx_memory = memory
                        ctx_steps_ago = _prev_steps_ago()
                    decision = prompt + memory_prompt(ctx_memory, steps_ago=ctx_steps_ago)
                    action_reply = 'ACTION: ' + action
                    turns.append((decision, action_reply))
                    if stats is not None:
                        stats['choice_action_stop' if has_stop else 'choice_action_move'] += 1
                    if is_exact_stop and stop_repeat > 1:
                        for _ in range(stop_repeat - 1):
                            turns.append((decision, action_reply))
                        if stats is not None:
                            stats['stop_repeat_extra'] += stop_repeat - 1

                for question, answer in turns:
                    out_samples.append({
                        "system": SYSTEM_PROMPT, "images": frames,
                        "messages": [{"from": "user", "value": question},
                                     {"from": "assistant", "value": answer}],
                        "episode_id": episode_id, "task type": "vln",
                    })
            continue

        memory = ''
        last_reason_step = None
        for start, action in chunks:
            frames = normalize_pair(sample_history_plus_current(images[:start + 1], max_history))
            prompt = build_vln_prompt(instruction, len(frames))
            turns = []
            if cot_index is None or action_control:
                turns.append((prompt, action))
            else:
                has_stop = 'stop' in action.lower()
                is_exact_stop = action.strip().lower() == 'stop'
                steps_ago = None if last_reason_step is None else start - last_reason_step
                if start in nodes:
                    decision = prompt + memory_prompt(memory, steps_ago=steps_ago)
                    turns.append((decision, 'REASON: ' + nodes[start]))
                    if stats is not None:
                        stats['choice_reason'] += 1
                    memory = nodes[start]
                    last_reason_step = start

                    action_prompt = prompt + memory_prompt(memory, action_only=True, steps_ago=0)
                    action_reply = 'ACTION: ' + action
                    turns.append((action_prompt, action_reply))
                    if stats is not None:
                        stats['action_after_reason'] += 1
                        if has_stop:
                            stats['action_after_reason_stop'] += 1
                    if is_exact_stop and stop_repeat > 1:
                        for _ in range(stop_repeat - 1):
                            turns.append((action_prompt, action_reply))
                        if stats is not None:
                            stats['stop_repeat_extra'] += stop_repeat - 1
                else:
                    if drop_pre_node_buffer and (start in pre_node_steps) and not has_stop:
                        if stats is not None:
                            stats['dropped_pre_node_buffer'] += 1
                        continue
                    if (not has_stop) and direct_action_keep_prob < 1.0 and rng.random() > direct_action_keep_prob:
                        if stats is not None:
                            stats['dropped_direct_move'] += 1
                        continue
                    decision = prompt + memory_prompt(memory, steps_ago=steps_ago)
                    action_reply = 'ACTION: ' + action
                    turns.append((decision, action_reply))
                    if stats is not None:
                        stats['choice_action_stop' if has_stop else 'choice_action_move'] += 1
                    if is_exact_stop and stop_repeat > 1:
                        for _ in range(stop_repeat - 1):
                            turns.append((decision, action_reply))
                        if stats is not None:
                            stats['stop_repeat_extra'] += stop_repeat - 1
            for question, answer in turns:
                out_samples.append({
                    "system": SYSTEM_PROMPT, "images": frames,
                    "messages": [{"from": "user", "value": question},
                                 {"from": "assistant", "value": answer}],
                    "episode_id": episode_id, "task type": "vln",
                })
    return missing


def main():
    parser = argparse.ArgumentParser(
        description="Convert R2R / RxR / ScaleVLN trajectory annotations into VLN SFT samples.",
    )
    parser.add_argument("--annotation_path", required=True,
                        help="JSONL (R2R/RxR) or JSON-array (ScaleVLN) of episode annotations.")
    parser.add_argument("--image_root", required=True,
                        help="Directory containing per-episode rendered frames.")
    parser.add_argument("--output_path", required=True,
                        help="Output JSON with the converted samples.")
    parser.add_argument("--relative_to", help="Write image paths relative to this dataset root.")
    parser.add_argument("--max_history", type=int, default=8, help="Number of history frames to sample (default: 8)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cot-index", type=Path, help="Combined reasoning exports/cot_index.json")
    parser.add_argument("--dataset", choices=("r2r", "rxr"))
    parser.add_argument("--action-control", action="store_true",
                        help="Use the same annotated trajectories and chunk boundaries with plain action targets")
    parser.add_argument("--drop-pre-node-buffer", action="store_true",
                        help="Skip non-stop only_act decision chunks at step = node_index - 1")
    parser.add_argument("--direct-action-keep-prob", type=float, default=1.0,
                        help="Keep probability in (0, 1] for non-stop only_act move chunks")
    parser.add_argument("--stop-repeat", type=int, default=1,
                        help="Number of copies to emit for exact 'ACTION: stop' samples")
    parser.add_argument("--soft-window", action="store_true",
                        help="Enable AwareVLN-style 5-frame trapezoidal soft-window reasoning sampling")
    parser.add_argument("--soft-window-pre-frames", type=int, default=1,
                        help="Frames before segment start_step to include in soft window (default: 1)")
    parser.add_argument("--empty-memory-drop-prob", type=float, default=0.0,
                        help="Probability in [0, 1) to drop memory on only_act frames in soft-window mode")
    args = parser.parse_args()
    if bool(args.cot_index) != bool(args.dataset) or (args.action_control and not args.cot_index):
        parser.error("--cot-index and --dataset are required together for annotated data")
    if not (0.0 < args.direct_action_keep_prob <= 1.0):
        parser.error("--direct-action-keep-prob must be in (0, 1]")
    if args.stop_repeat < 1:
        parser.error("--stop-repeat must be >= 1")
    if args.soft_window_pre_frames < 0:
        parser.error("--soft-window-pre-frames must be >= 0")
    if not (0.0 <= args.empty_memory_drop_prob < 1.0):
        parser.error("--empty-memory-drop-prob must be in [0, 1)")
    if (args.drop_pre_node_buffer or args.direct_action_keep_prob < 1.0 or args.stop_repeat != 1
            or args.soft_window or args.empty_memory_drop_prob > 0.0) and (
            not args.cot_index or args.action_control):
        parser.error("Reasoning sampling flags require --cot-index without --action-control")
    cot_index = json.loads(args.cot_index.read_text()) if args.cot_index else None

    global MAX_HISTORY
    if args.max_history < 1:
        parser.error("--max_history must be positive")
    MAX_HISTORY = args.max_history

    random.seed(args.seed)
    sampling_rng = random.Random(args.seed + 1000003)

    episodes = load_episodes(args.annotation_path)

    samples = []
    missing = 0
    stats = Counter()
    for episode in tqdm(episodes, desc="vln"):
        missing += convert_episode_vln(
            episode,
            args.image_root,
            samples,
            cot_index=cot_index,
            dataset=args.dataset,
            action_control=args.action_control,
            drop_pre_node_buffer=args.drop_pre_node_buffer,
            direct_action_keep_prob=args.direct_action_keep_prob,
            stop_repeat=args.stop_repeat,
            soft_window=args.soft_window,
            soft_window_pre_frames=args.soft_window_pre_frames,
            empty_memory_drop_prob=args.empty_memory_drop_prob,
            sampling_rng=sampling_rng,
            stats=stats,
        ) or 0
    if cot_index is not None:
        print(f"Skipped {missing} instructions missing from the completed reasoning index")
        if not args.action_control:
            decision_total = (
                stats['choice_reason']
                + stats['choice_action_move']
                + stats['choice_action_stop']
            )
            reason_pct = (100.0 * stats['choice_reason'] / decision_total) if decision_total else 0.0
            print(
                "Reasoning sample stats: "
                f"choice_reason={stats['choice_reason']} ({reason_pct:.1f}% of base decisions), "
                f"choice_action_move={stats['choice_action_move']}, "
                f"choice_action_stop={stats['choice_action_stop']}, "
                f"action_after_reason={stats['action_after_reason']} "
                f"(stop={stats['action_after_reason_stop']}), "
                f"dropped_pre_node_buffer={stats['dropped_pre_node_buffer']}, "
                f"dropped_direct_move={stats['dropped_direct_move']}, "
                f"stop_repeat_extra={stats['stop_repeat_extra']}, "
                f"empty_memory_dropped={stats['empty_memory_dropped']}, "
                f"total_emitted={len(samples)}"
            )

    if not samples:
        parser.error("No valid samples were produced; check annotations and frame paths")
    try:
        relativize_image_paths(samples, args.relative_to)
    except ValueError as exc:
        parser.error(str(exc))
    write_samples(args.output_path, samples)
    print(f"wrote {len(samples)} samples -> {args.output_path}")


if __name__ == "__main__":
    main()

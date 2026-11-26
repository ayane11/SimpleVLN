#!/usr/bin/env python3
"""Quick 1-minute offline vLLM probe before launching full Habitat eval.

Tests unseen R2R episodes from v2 annotations on:
1. 50 REASON nodes -> REASON recall
2. 50 non-node move frames -> REASON false-positive rate & first-action match
3. 50 STOP frames -> exact/contains stop accuracy under orig / emptymem / realmem
   (simulating two-round REASON -> action_only=True just like eval.py).
"""

import argparse
import base64
from collections import Counter
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
RUNS = Path('/workspace/VLN/runs')
sys.path.insert(0, str(ROOT))
from navigation_memory import memory_prompt, parse_reply  # noqa: E402


def b64_url(path: str) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def build_messages(system_prompt: str, user_text: str, image_paths: list[str]) -> list[dict]:
    parts = user_text.split("<image>")
    if len(parts) != len(image_paths) + 1:
        raise ValueError(f"Expected {len(image_paths)} <image> tags, got {len(parts) - 1}")
    content = []
    for i, part in enumerate(parts):
        if part:
            content.append({"type": "text", "text": part})
        if i < len(image_paths):
            content.append({"type": "image_url", "image_url": {"url": b64_url(image_paths[i])}})
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": content},
    ]


def call_vllm(base_url: str, model: str, messages: list[dict], max_tokens: int = 128) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=60) as resp:
        data = json.loads(resp.read().decode())
    return (data["choices"][0]["message"]["content"] or "").strip()


def first_action_kind(text: str) -> str:
    s = text.strip()
    if s.startswith("ACTION:"):
        s = s[7:].strip()
    s = s.split(",")[0].strip().lower()
    if "stop" in s:
        return "stop"
    if "forward" in s:
        return "forward"
    if "left" in s:
        return "left"
    if "right" in s:
        return "right"
    return "other"


def split_base_prompt(user_text: str) -> str:
    return user_text.split("\nNavigation memory")[0]


def pick_unseen(rows: list[dict], train_eps: set[str], limit: int):
    reason_rows, move_rows, stop_rows = [], [], []
    seen_r, seen_m, seen_s = set(), set(), set()
    for row in rows:
        ep = str(row["episode_id"])
        if ep in train_eps:
            continue
        u, a = row["messages"][0]["value"], row["messages"][1]["value"]
        if a.startswith("REASON:") and ep not in seen_r and len(reason_rows) < limit:
            reason_rows.append(row)
            seen_r.add(ep)
        elif a == "ACTION: stop" and "Choose whether" in u and ep not in seen_s and len(stop_rows) < limit:
            stop_rows.append(row)
            seen_s.add(ep)
        elif (a.startswith("ACTION:") and "stop" not in a.lower()
              and "Choose whether" in u and ep not in seen_m and len(move_rows) < limit):
            move_rows.append(row)
            seen_m.add(ep)
        if len(reason_rows) == limit and len(move_rows) == limit and len(stop_rows) == limit:
            break
    return reason_rows, move_rows, stop_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:22012/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--annotations", type=Path,
                   default=RUNS / "outputs/sft/reasoning_sft_4b_v4/annotations/SFT/R2R.json")
    p.add_argument("--split", type=Path, default=None)
    p.add_argument("--limit", type=int, default=50)
    args = p.parse_args()

    rows = json.loads(args.annotations.read_text())
    if args.split is not None and args.split.exists():
        split_data = json.loads(args.split.read_text())
        train_eps = {str(e) for e in split_data["datasets"]["R2R"]["episode_ids"]}
    else:
        train_eps = set()
    reason_rows, move_rows, stop_rows = pick_unseen(rows, train_eps, args.limit)
    print(f"Probing unseen R2R: reason={len(reason_rows)}, move={len(move_rows)}, stop={len(stop_rows)}", flush=True)

    # 1. REASON recall on true nodes
    r_pred, r_invalid = 0, 0
    for row in reason_rows:
        out = call_vllm(args.base_url, args.model,
                        build_messages(row["system"], row["messages"][0]["value"], row["images"]))
        try:
            mode, _ = parse_reply(out)
        except ValueError:
            r_invalid += 1
            continue
        if mode == "REASON":
            r_pred += 1
    print(f"[REASON nodes] recall={r_pred}/{len(reason_rows)} ({100*r_pred/max(1,len(reason_rows)):.1f}%), invalid={r_invalid}", flush=True)

    # 2. Non-node move frames
    m_fp, m_match, m_invalid = 0, 0, 0
    for row in move_rows:
        base_u = split_base_prompt(row["messages"][0]["value"])
        out = call_vllm(args.base_url, args.model,
                        build_messages(row["system"], row["messages"][0]["value"], row["images"]))
        try:
            mode, body = parse_reply(out)
        except ValueError:
            m_invalid += 1
            continue
        if mode == "REASON":
            m_fp += 1
            u2 = base_u + memory_prompt(body, action_only=True, steps_ago=0)
            out2 = call_vllm(args.base_url, args.model, build_messages(row["system"], u2, row["images"]), max_tokens=48)
            try:
                _, body = parse_reply(out2, action_only=True)
            except ValueError:
                m_invalid += 1
                continue
        if first_action_kind(body) == first_action_kind(row["messages"][1]["value"]):
            m_match += 1
    print(f"[Non-node move] false_pos_reason={m_fp}/{len(move_rows)} ({100*m_fp/max(1,len(move_rows)):.1f}%), first_act_match={m_match}/{len(move_rows)}, invalid={m_invalid}", flush=True)

    # 3. STOP frames
    for tag in ("orig", "emptymem_choice", "realmem_choice"):
        exact_stop, has_stop, triggered_reason, invalid = 0, 0, 0, 0
        samples = []
        for row in stop_rows:
            base_u = split_base_prompt(row["messages"][0]["value"])
            if tag == "orig":
                u = base_u
            elif tag == "emptymem_choice":
                u = base_u + memory_prompt("")
            else:
                u = row["messages"][0]["value"]
            out = call_vllm(args.base_url, args.model,
                            build_messages(row["system"], u, row["images"]), max_tokens=128)
            if tag == "orig":
                body = out
            else:
                try:
                    mode, body = parse_reply(out)
                except ValueError:
                    invalid += 1
                    continue
                if mode == "REASON":
                    triggered_reason += 1
                    u2 = base_u + memory_prompt(body, action_only=True, steps_ago=0)
                    out2 = call_vllm(args.base_url, args.model, build_messages(row["system"], u2, row["images"]), max_tokens=48)
                    try:
                        _, body = parse_reply(out2, action_only=True)
                    except ValueError:
                        invalid += 1
                        continue
            samples.append(body)
            b_low = body.strip().lower()
            if b_low.startswith("stop"):
                exact_stop += 1
            if "stop" in b_low:
                has_stop += 1
        print(f"[STOP {tag:15s}] exact={exact_stop}/{len(stop_rows)} ({100*exact_stop/max(1,len(stop_rows)):.1f}%), has_stop={has_stop}/{len(stop_rows)}, reason_triggered={triggered_reason}, invalid={invalid} | ex={samples[:3]}", flush=True)


if __name__ == "__main__":
    main()

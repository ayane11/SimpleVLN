"""Text protocol shared by reasoning SFT and closed-loop navigation."""

import re


_STALE_NEXT_PLAN_RE = re.compile(
    r'(?:\s+The next plan is to (?:move forward|turn left|turn right|stop and wait)\.)+\s*$'
)


def strip_stale_next_plan(memory: str) -> str:
    """Strip trailing immediate-action sentence from historical navigation memory."""
    cleaned = memory.strip()
    if not cleaned:
        return ''
    stripped = _STALE_NEXT_PLAN_RE.sub('', cleaned).strip()
    return stripped or cleaned


def memory_prompt(memory, action_only=False, steps_ago=None, strip_stale_plan=True):
    if steps_ago is not None and steps_ago < 0:
        raise ValueError(f"steps_ago must be non-negative, got {steps_ago!r}")
    task = ("Return ACTION: followed by the next action chunk." if action_only else
            "Choose whether to update your navigation memory or act. "
            "Return REASON: followed by the current scene, task progress and next plan, "
            "or ACTION: followed by the next action chunk.")
    mem_raw = (memory or "").strip()
    if not mem_raw:
        header = "Navigation memory: (none)"
    else:
        if strip_stale_plan and steps_ago is not None and steps_ago > 0:
            mem_text = strip_stale_next_plan(mem_raw)
        else:
            mem_text = mem_raw
        if steps_ago is None:
            header = f"Navigation memory: {mem_text}"
        else:
            unit = "step" if steps_ago == 1 else "steps"
            header = f"Navigation memory (updated {steps_ago} {unit} ago): {mem_text}"
    return f"\n{header}\n{task}"


def parse_reply(text, action_only=False):
    mode, separator, body = text.strip().partition(':')
    body = body.strip()
    if not separator or not body or mode not in (('ACTION',) if action_only else ('ACTION', 'REASON')):
        raise ValueError(f"Invalid navigation reply: {text[:120]!r}")
    return mode, body

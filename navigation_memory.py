"""Text protocol shared by reasoning SFT and closed-loop navigation."""

import re


_STALE_NEXT_PLAN_RE = re.compile(
    r'(?:\s+The next plan is to (?:move forward|turn left|turn right|stop and wait)\.)+\s*$'
)
_STANDALONE_NEXT_PLAN_RE = re.compile(
    r'^(?:The next plan is to (?:move forward|turn left|turn right|stop and wait)\.\s*)+$'
)
_TRAILING_NEXT_PLAN_CAPTURE_RE = re.compile(
    r'(?:^|\s+)(The next plan is to (?:move forward|turn left|turn right|stop and wait)\.)'
    r'(?:\s+The next plan is to (?:move forward|turn left|turn right|stop and wait)\.)*\s*$'
)

_NEGATION_GUARD_RE = re.compile(
    r"\b(?:not yet|has not|have not|haven't|hasn't|ambiguous|unclear|cannot|can't)\b",
    re.IGNORECASE,
)

_ACTION_TARGET = (
    r'(?:turn|turning|move|moving|walk|walking|go|going|proceed|proceeding|'
    r'continue|continuing|head|heading|step|stepping|enter|entering|exit|exiting|'
    r'ascend|ascending|descend|descending|climb|climbing|wait|waiting|stop|stopping|'
    r'orient|orienting|adjust|adjusting|align|aligning|approach|approaching|'
    r'make\s+(?:a|an|the)|making\s+(?:a|an|the)|take\s+(?:a|an|the)|taking\s+(?:a|an|the)|'
    r'execute\b|executing\b|follow\b|following\b|perform\b|performing\b|'
    r'complete\s+the\s+(?:wait|stop|turn)\b)'
)

_PRESCRIPTIVE_SENT_RE = re.compile(
    r'^(?:'
    r'The next (?:immediate |required )?(?:step|move|action|required action|instruction|subtask|part of the instruction)\s+is\b.*|'
    r'The instruction\b.*?\bis the next (?:immediate |required )?(?:step|move|action|required action)\b.*|'
    r'(?:Now,?\s*|Next,?\s*)?(?:the agent|I)\s+(?:is\s+|am\s+)?(?:currently\s+|now\s+)?'
    r'(?:should|must|needs? to|positioned to|ready to|about to|preparing to)\s+'
    r'(?:now\s+|begin\s+|start\s+)?' + _ACTION_TARGET + r'.*|'
    r'(?:The agent|I)\s+(?:is|am)\s+(?:currently\s+|now\s+)?(?:facing|standing|positioned|at|in|inside|near)\b.*?'
    r'(?:,\s*|\s+and\s+)(?:is\s+|am\s+)?(?:currently\s+|now\s+)?'
    r'(?:needs? to|needing to|should|must|preparing to|ready to|about to|positioned to)\s+'
    r'(?:now\s+|begin\s+|start\s+)?' + _ACTION_TARGET + r'.*'
    r')$',
    re.IGNORECASE,
)

_PRESCRIPTIVE_CLAUSE_RE = re.compile(
    r'(?:,\s+and\s+(?:the agent|I)\s+(?:is\s+|am\s+)?(?:currently\s+|now\s+)?'
    r'(?:should|must|needs? to|positioned to|ready to|about to|preparing to)\s+'
    r'(?:now\s+|begin\s+|start\s+)?' + _ACTION_TARGET + r'[^.!?]*'
    r'|,\s*(?:needing to|preparing to|ready to|about to|positioning itself to)\s+'
    r'(?:now\s+|begin\s+|start\s+)?' + _ACTION_TARGET + r'[^.!?]*'
    r'|\s+and\s+(?:is\s+|am\s+)?(?:needs? to|preparing to|ready to|about to)\s+'
    r'(?:now\s+|begin\s+|start\s+)?' + _ACTION_TARGET + r'[^.!?]*'
    r')([.!?])$',
    re.IGNORECASE,
)

_PROGRESS_INDICATOR_RE = re.compile(
    r'\b(?:the agent|I\s+(?:have|am)|instruction|subtask|entered|exited|moved|navigated|walked|turned|'
    r'reached|arrived|completed|passed|crossed|ascended|descended|approaching|standing|positioned)\b',
    re.IGNORECASE,
)


def _clean_progress_pass(body: str) -> str:
    text = body.strip()
    if not text:
        return ''
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if len(sents) >= 2 and not _NEGATION_GUARD_RE.search(sents[-1]) and _PRESCRIPTIVE_SENT_RE.match(sents[-1]):
        non_scene = [s for s in sents[:-1] if not s.startswith('I see ')]
        if non_scene and _PROGRESS_INDICATOR_RE.search(' '.join(non_scene)):
            sents.pop()
            return ' '.join(sents)
    if sents and not _NEGATION_GUARD_RE.search(sents[-1]):
        new_last, n = _PRESCRIPTIVE_CLAUSE_RE.subn(r'\1', sents[-1])
        if n > 0 and len(new_last.split()) >= 6:
            sents[-1] = new_last.strip()
            return ' '.join(sents)
    return ' '.join(sents)


def clean_stale_progress_tail(body: str) -> str:
    """Remove redundant immediate-action prescriptions at the end of progress text."""
    current = body.strip()
    for _ in range(2):
        updated = _clean_progress_pass(current)
        if updated == current:
            break
        current = updated
    return current


def action_plan_sentence(action: str) -> str:
    """Map an atomic or chunked navigation action string to the canonical next-plan sentence."""
    first = (action or '').split(',')[0].strip().lower()
    if first == 'stop':
        return 'The next plan is to stop and wait.'
    if first.startswith('forward') or first.startswith('move forward'):
        return 'The next plan is to move forward.'
    if first.startswith('turn left'):
        return 'The next plan is to turn left.'
    if first.startswith('turn right'):
        return 'The next plan is to turn right.'
    raise ValueError(f"Unsupported action for next-plan sentence: {action!r}")


def clean_reasoning_update(memory: str) -> str:
    """Clean redundant prescriptive progress tails while preserving the canonical next-plan sentence."""
    cleaned = (memory or '').strip()
    if not cleaned:
        return ''
    match = _TRAILING_NEXT_PLAN_CAPTURE_RE.search(cleaned)
    if not match:
        return clean_stale_progress_tail(cleaned)
    body = clean_stale_progress_tail(cleaned[:match.start()].strip())
    plan = match.group(1)
    return f"{body} {plan}".strip() if body else plan


def rewrite_next_plan(memory: str, action: str) -> str:
    """Align the canonical trailing next-plan sentence with the current frame's expert action."""
    cleaned = (memory or '').strip()
    if not cleaned:
        return ''
    new_plan = action_plan_sentence(action)
    if _STANDALONE_NEXT_PLAN_RE.match(cleaned):
        return new_plan
    match = _TRAILING_NEXT_PLAN_CAPTURE_RE.search(cleaned)
    if not match:
        return clean_stale_progress_tail(cleaned)
    body = clean_stale_progress_tail(cleaned[:match.start()].strip())
    return f"{body} {new_plan}".strip() if body else new_plan


def strip_stale_next_plan(memory: str) -> str:
    """Strip trailing immediate-action sentences from historical navigation memory."""
    cleaned = memory.strip()
    if not cleaned:
        return ''
    if _STANDALONE_NEXT_PLAN_RE.match(cleaned):
        return cleaned
    stripped = _STALE_NEXT_PLAN_RE.sub('', cleaned).strip()
    stripped = clean_stale_progress_tail(stripped)
    return stripped or cleaned


_ATOMIC_ACTION_PAT = (
    r'(?:stop|(?:move\s+)?forward(?:\s+-?\d+(?:\.\d+)?(?:\s*cm)?)?|'
    r'(?:turn\s+)?(?:left|right)(?:\s+-?\d+(?:\.\d+)?(?:\s*degrees?)?)?)'
)
_PLAIN_ACTION_CHUNK_RE = re.compile(
    rf'^(?:<answer>\s*)?{_ATOMIC_ACTION_PAT}(?:\s*,\s*{_ATOMIC_ACTION_PAT})*(?:\s*</answer>)?\s*\.?$',
    re.IGNORECASE,
)


def memory_prompt(memory, action_only=False, steps_ago=None, strip_stale_plan=True):
    if steps_ago is not None and steps_ago < 0:
        raise ValueError(f"steps_ago must be non-negative, got {steps_ago!r}")
    task = ("Return the next action chunk." if action_only else
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
    cleaned = text.strip()
    if not cleaned:
        raise ValueError(f"Invalid navigation reply: {text[:120]!r}")
    if action_only:
        mode, separator, body = cleaned.partition(':')
        if separator:
            body = body.strip()
            if mode.strip() != 'ACTION' or not body or not _PLAIN_ACTION_CHUNK_RE.match(body):
                raise ValueError(f"Invalid navigation reply: {text[:120]!r}")
            return 'ACTION', body
        if not _PLAIN_ACTION_CHUNK_RE.match(cleaned):
            raise ValueError(f"Invalid navigation reply: {text[:120]!r}")
        return 'ACTION', cleaned
    mode, separator, body = cleaned.partition(':')
    body = body.strip()
    if not separator or not body or mode not in ('ACTION', 'REASON'):
        raise ValueError(f"Invalid navigation reply: {text[:120]!r}")
    return mode, body

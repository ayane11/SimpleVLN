"""Opt-in structured teacher constraints. Checks are not semantic verification."""
import json
import re
from core import validate_reason, reason_text


def tokens(text):
    return re.findall(r"\w+", text.lower())


def decomposition_prompt(instruction):
    return ('Split this navigation instruction into ordered short subtasks/clauses. '
            'Return JSON {"subtasks":["verbatim first span", "verbatim next span", ...]}. '
            'Use consecutive original text spans; cover every word exactly once in original order. '
            'Do not paraphrase, add, omit, or resolve ambiguous/self-corrected instructions. '
            'Keep each movement or landmark goal separate when possible. Only segment text, '
            'do not judge completion or view any future observations.\nInstruction: '+json.dumps(instruction))


def validate_subtasks(value, instruction):
    spans=value.get('subtasks')
    if not isinstance(spans,list) or not spans or any(not isinstance(s,str) or not s.strip() for s in spans):
        raise ValueError('subtasks must be nonempty verbatim spans')
    if tokens(' '.join(spans)) != tokens(instruction):
        raise ValueError('Subtasks must preserve EVERY original word exactly once, in original order; no paraphrasing')


def action_kind(action):
    if action == 'stop': return 'stop'
    if action.startswith('forward'): return 'forward'
    if action.startswith('turn left'): return 'left'
    if action.startswith('turn right'): return 'right'
    raise ValueError('Unsupported expert action: '+action)


def action_sentence(action):
    return {'forward':'The next plan is to move forward.',
            'left':'The next plan is to turn left.',
            'right':'The next plan is to turn right.',
            'stop':'The next plan is to stop and wait.'}[action_kind(action)]


def action_runs(actions):
    runs=[]
    for action in actions:
        if runs and runs[-1]['action']==action: runs[-1]['count']+=1
        else: runs.append(dict(action=action,count=1))
    return runs


def terminal_stop_ids(subtasks):
    """Conservative final stop/wait clause detection, not intermediate pauses."""
    found=[i for i,s in enumerate(subtasks,1) if re.search(r'\b(?:stop|wait)\b',s,re.I)]
    if not found:return []
    last=found[-1]
    if any(re.search(r'\b(?:walk|move|turn|enter|exit|climb|go)\b',s,re.I) for s in subtasks[last:]):return []
    return [last]


def progress_prompt(instruction, node, actions, history, subtasks, previous_state, previous_step):
    t=node['step'];start=previous_step if previous_step is not None else 0
    return f'''Annotate navigation at step {t}, BEFORE action {t}. The ONLY current image is step {t}.
Instruction: {json.dumps(instruction)}
Ordered instruction spans (one-based IDs): {json.dumps(dict(enumerate(subtasks,1)))}
Approximate room metadata: {json.dumps(node)}
Visible causal frame IDs: {history}
Previous structured progress (fallible, use as memory): {json.dumps(previous_state)}
Already executed actions from step {start} inclusive to {t} exclusive, run-length encoded:
{json.dumps(action_runs(actions[start:t]))}
CURRENT expert action: {json.dumps(actions[t])}
Upcoming expert actions (not yet executed): {json.dumps(actions[t:t+3])}
Only CURRENT STOP permits stop_now=true or task_complete=true. A future STOP does NOT permit stopping now.
Terminal stop/wait subtask IDs: {terminal_stop_ids(subtasks)}. These must NEVER be completed before CURRENT STOP;
arriving at a room or standing near a goal is not the same as completing its stop/wait instruction.
Current expert action controls only the immediate movement, not semantic completion.
Keep next_plan EXACTLY {json.dumps(action_sentence(actions[t]))}. This canonical immediate-action sentence
is deliberately constrained, not a free-form long-term plan. Do not append any future actions.
Describe current visible objects in scene_description, not historical objects or instruction landmarks.
Update progress using current/recent images, executed actions and retained supported memory. Unknown
means unverified, NOT not completed. Never say 'I have not' or 'not yet' solely because a milestone
is absent from recent images. By default report only progress supported by the observations and memory.
For PROGRESS, examine ALL supplied historical frames and previous state, not only the current frame.
The current-only restriction applies to scene_description, NOT to progress_assessment. Do not cite
'not visible in the current view' as the reason to forget an event shown in history. Summarize the
latest supported local progress in one or two short sentences. Mention ambiguity in progress_assessment
only if it changes whether the current subtask can be marked complete. In that case identify the
specific observation that leaves completion ambiguous. Leave unrelated or future milestones out
of the prose. Put unresolved older milestones in
uncertainties and unknown states, rather than appending a long list of doubts to every target.
Do not reset a previously completed subtask just because it is now out of view. To correct an old
completed status, give a correction with fresh visible evidence and a reason. A newly completed
subtask needs evidence from images supplied NOW, not just future expert actions or the instruction.
Do not force all subtasks complete at STOP: the expert trajectory and instruction may disagree.
Sparse observations may leave earlier spans unknown. Do not fabricate intermediate achievements.
progress_assessment must be consistent with the state; state only supported completed/local progress.
Avoid references to annotators, expert hints, ground truth or this JSON in the three target sentences.
Return JSON with these fields:
{{"scene_description":"I see ...", "progress_assessment":"<brief evidence-based local progress>",
"next_plan":{json.dumps(action_sentence(actions[t]))}, "evidence_steps":[{t}], "uncertainties":[],
"immediate_action":"{action_kind(actions[t])}", "stop_now":{str(actions[t]=='stop').lower()}, "task_complete":false,
"progress_state":[{{"id":1,"status":"unknown|in_progress|completed","evidence_steps":[]}}],
"active_subtask":1, "corrections":[{{"id":1,"reason":"why old completion was wrong","evidence_steps":[{t}]}}]}}
Include EVERY subtask once, in order. Use [] for no corrections. Use null for active_subtask when
no active subtask can be established; otherwise it must be an in_progress subtask. Unknown states
may have no evidence; other states need cited evidence. Previously unchanged statuses may retain
their earlier evidence IDs. Cite current frame {t} in the top-level evidence_steps.
'''


def validate_progress(value, history, current_action, subtasks, previous_state=None):
    validate_reason(value,history)
    if max(history) not in value['evidence_steps']: raise ValueError('Cite CURRENT frame at top level')
    expected=action_kind(current_action)
    if value.get('immediate_action')!=expected: raise ValueError('immediate_action must match CURRENT expert action '+expected)
    if type(value.get('stop_now')) is not bool or value['stop_now']!=(expected=='stop'):
        raise ValueError('stop_now must match CURRENT STOP, never a future STOP')
    if type(value.get('task_complete')) is not bool or (value['task_complete'] and expected!='stop'):
        raise ValueError('task_complete cannot be true before CURRENT STOP')
    if value['next_plan'].strip()!=action_sentence(current_action):
        raise ValueError('next_plan must be EXACTLY '+action_sentence(current_action))
    if re.search(r'\b(?:expert|ground truth|annotator)\b',reason_text(value),re.I):
        raise ValueError('Do not include privileged annotation references in target text')
    progress=value['progress_assessment'].lower()
    # Negation alone is not an error: "not yet at the final goal" may be supported.
    # Whether a negative claim follows from the images needs semantic review.
    if expected!='stop' and re.search(r'\b(?:completed|finished)\s+(?:(?:all|the)\s+)*(?:task|instructions?|route|navigation)\b|\b(?:reached|arrived at)\s+(?:the\s+)?(?:final destination|final goal|final stopping point|destination)\b',progress):
        raise ValueError('Overall completion claim before CURRENT STOP')
    states=value.get('progress_state')
    if not isinstance(states,list) or [s.get('id') for s in states if isinstance(s,dict)]!=list(range(1,len(subtasks)+1)):
        raise ValueError('progress_state must include every ordered subtask ID exactly once')
    old={s['id']:s for s in (previous_state or [])}
    corrections=value.get('corrections')
    if not isinstance(corrections,list): raise ValueError('corrections must be a list')
    correction_ids=set()
    for c in corrections:
        if not isinstance(c,dict) or c.get('id') not in old or c['id'] in correction_ids or not isinstance(c.get('reason'),str) or not c['reason'].strip():
            raise ValueError('Invalid correction ID or reason')
        evidence=c.get('evidence_steps')
        if not isinstance(evidence,list) or not evidence or any(type(e) is not int or e not in history for e in evidence):
            raise ValueError('Corrections need fresh supplied image evidence')
        correction_ids.add(c['id'])
    for s in states:
        status=s.get('status');evidence=s.get('evidence_steps');prior=old.get(s['id'])
        if status not in ['unknown','in_progress','completed'] or not isinstance(evidence,list):raise ValueError('Invalid subtask status/evidence')
        allowed=set(history)
        if prior:allowed.update(prior['evidence_steps'])
        if any(type(e) is not int or e not in allowed for e in evidence):raise ValueError('New status requires fresh visible evidence')
        if status=='completed' and (not prior or prior['status']!='completed') and not any(e in history for e in evidence):
            raise ValueError('New completion requires at least one currently supplied evidence frame; retained historical evidence may accompany it')
        if status!='unknown' and not evidence:raise ValueError('Supported statuses need image evidence')
        if expected!='stop' and s['id'] in terminal_stop_ids(subtasks) and status=='completed':
            raise ValueError('Terminal stop/wait subtask cannot be completed before CURRENT STOP')
        if prior and prior['status']=='completed' and status!='completed' and s['id'] not in correction_ids:
            raise ValueError('Cannot reset a completed subtask without an evidenced correction')
    active=value.get('active_subtask')
    if active is not None and (type(active) is not int or active not in range(1,len(subtasks)+1) or states[active-1]['status']!='in_progress'):
        raise ValueError('active_subtask must be null or an in_progress subtask')
    if value['task_complete'] and any(s['status']!='completed' for s in states):
        raise ValueError('Overall completion requires every subtask supported as completed')

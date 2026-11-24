"""Failure-only generator using the original manifest and successful request cache."""
import copy
import json
import re
from pathlib import Path

import generate

original_request = generate.request_json


def boundary_request(instruction):
    words = instruction.split()
    prompt = ('Choose inclusive end word numbers for consecutive navigation subtasks. '
              'Return only {"ends":[...]}, strictly increasing integers. '
              f'The last must be {len(words)}. Separate distinct movements and stop clauses.\n'
              + ' '.join(f'{i}:{w}' for i, w in enumerate(words, 1)))

    def decode(raw):
        ends = raw.get('ends')
        if not isinstance(ends, list) or not ends or any(type(i) is not int for i in ends):
            raise ValueError('ends must be a nonempty list of integers')
        spans = re.findall(r'\S+\s*', instruction)
        start = 0
        result = []
        for end in ends:
            if not start < end <= len(words):
                raise ValueError(f'Choose increasing boundaries within 1..{len(words)}')
            result.append(''.join(spans[start:end]).strip())
            start = end
        if start != len(words):
            raise ValueError(f'Last boundary must be {len(words)}')
        return {'subtasks': result}

    return prompt, decode


def frame_request(content):
    labels = {}
    for item in content:
        match = re.fullmatch(r'causal history/current: step=(\d+)', item.get('text', ''))
        if match:
            step = int(match[1])
            tag = f'STEP_{step:03d}'
            labels[tag] = step
            item['text'] += f' | image tag {tag}'
    prefix = 'Previous structured progress (fallible, use as memory): '
    previous = json.loads(next(line[len(prefix):] for line in content[0]['text'].splitlines()
                               if line.startswith(prefix))) or []
    prior = {s['id']: s for s in previous}
    content.append({'type': 'text', 'text': (
        'Evidence format override: all evidence_steps arrays use exact image tag strings. '
        f'Allowed tags: {list(labels)}. Never use integers or image positions. '
        'For an unchanged subtask you may return "KEEP" in place of its evidence array '
        'to retain its own prior evidence. Use unknown and [] if unsupported. '
        'Cite the current tag at top level. Only describe observed progress and visible landmarks.')})

    def decode(raw):
        value = copy.deepcopy(raw)
        def convert(row, old=None):
            refs = row.get('evidence_steps')
            if refs == 'KEEP' and old and row.get('status') == old['status']:
                row['evidence_steps'] = old['evidence_steps']
            elif isinstance(refs, list) and all(isinstance(x,str) and x in labels for x in refs):
                row['evidence_steps'] = [labels[x] for x in refs]
            else:
                raise ValueError(f'Invalid evidence {refs!r} for ID {row.get("id")}; choose from {list(labels)}')
        convert(value)
        for s in value['progress_state']:
            convert(s, prior.get(s['id']))
        for c in value['corrections']:
            convert(c)
        return value
    return decode


def repair_request(teacher, messages, validator, settings, output, retries):
    images = [generate.file_identity(c['image']) for m in messages if isinstance(m['content'], list)
              for c in m['content'] if c['type'] == 'image']
    key = generate.digest(dict(messages=messages, images=images, settings=settings))
    if (output/'requests'/f'{key}.json').exists():
        return original_request(teacher, messages, validator, settings, output, retries)
    messages = copy.deepcopy(messages)
    content = messages[-1]['content']
    if isinstance(content, str):
        instruction = json.loads(content.rsplit('\nInstruction: ', 1)[1])
        messages[-1]['content'], decode = boundary_request(instruction)
    else:
        decode = frame_request(content)
    profile = dict(settings, repair_code_hash=generate.digest(Path(__file__).read_text()))
    raw, key = original_request(teacher, messages, lambda v: validator(decode(v)), profile, output, retries)
    return decode(raw), key


if __name__ == '__main__':
    generate.request_json = repair_request
    generate.main()

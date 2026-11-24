"""Conservative room events from reviewed, aligned simulator replay records.

Input: {alignment_status: 'approved', steps: [{step, region_candidates,
traveled_m, progress_ratio, ...}]}. Unknown states are never guessed.
"""
import argparse
import json
from pathlib import Path
from core import digest

RULE_VERSION = 'room-category-confirm-2-v1'


def category(candidates):
    names = {str(r.get('category') or '').strip().lower() for r in candidates}
    if not names or names & {'', 'unknown', 'none', 'unlabeled', 'undefined'}:
        return None
    return next(iter(names)) if len(names) == 1 else None


def detect(steps):
    confirmed = pending = None
    count = 0
    unknown_gap = False
    events = []
    for i, row in enumerate(steps):
        if row['step'] != i:
            raise ValueError('Replay steps must be contiguous zero-based pre-action observations')
        current = category(row['region_candidates'])
        if current is None:
            pending, count = None, 0
            unknown_gap |= confirmed is not None
            continue
        count = count + 1 if current == pending else 1
        pending = current
        if count < 2:
            continue
        if confirmed is not None and current != confirmed:
            events.append(dict(step=i, type='newroom', room_from=confirmed,
                               room_to=current, confirmation_steps=[i-1, i],
                               crossed_unknown=unknown_gap,
                               region_candidates=row['region_candidates'],
                               progress_ratio=row.get('progress_ratio'),
                               traveled_m=row.get('traveled_m'), rule_version=RULE_VERSION,
                               event_evidence='Two consecutive unambiguous simulator room categories'))
        confirmed = current
        unknown_gap = False
    return events


def export(records):
    events, excluded = {}, []
    for record in records:
        eid = str(record['episode_id'])
        if record.get('alignment_status') != 'approved':
            excluded.append(eid)
            continue
        if eid in events:
            raise ValueError('Duplicate episode ID; export datasets separately')
        events[eid] = detect(record['steps'])
    return dict(events=events, excluded=excluded, rule_version=RULE_VERSION,
                source_hash=digest(records))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replays', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = export(json.loads(Path(args.replays).read_text()))
    Path(args.output).write_text(json.dumps(result['events'], indent=2))
    Path(args.output + '.audit.json').write_text(json.dumps(result, indent=2))

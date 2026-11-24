#!/usr/bin/env python3
"""Clean redundant immediate-action prescriptions from exported reasoning segments."""

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from navigation_memory import clean_reasoning_update  # noqa: E402


def atomic_write_json(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', dir=path.parent, prefix=f'.{path.name}.', delete=False
        ) as handle:
            tmp_name = handle.name
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
        os.replace(tmp_name, path)
    finally:
        if tmp_name is not None and os.path.exists(tmp_name):
            os.unlink(tmp_name)


def backup_once(path: Path) -> Path | None:
    if not path.is_file():
        return None
    backup = path.with_suffix('.uncleaned' + path.suffix)
    if not backup.exists():
        tmp = backup.with_name(f'.{backup.name}.tmp')
        shutil.copy2(path, tmp)
        os.replace(tmp, backup)
    return backup


def clean_segments(segments: list[dict]) -> tuple[list[dict], int, int]:
    """Clean updated_content on reasoning nodes and propagate to subsequent content fields."""
    out = []
    previous_cleaned = None
    total_nodes = 0
    changed_nodes = 0
    for seg in segments:
        item = copy.deepcopy(seg)
        if item.get('type') != 'only_act' and isinstance(item.get('updated_content'), str):
            total_nodes += 1
            if item.get('content') is not None:
                item['content'] = previous_cleaned if previous_cleaned is not None else clean_reasoning_update(item['content'])
            cleaned = clean_reasoning_update(item['updated_content'])
            if cleaned != item['updated_content'].strip():
                changed_nodes += 1
            item['updated_content'] = cleaned
            previous_cleaned = cleaned
        else:
            if item.get('content') is not None:
                item['content'] = previous_cleaned if previous_cleaned is not None else clean_reasoning_update(item['content'])
        out.append(item)
    return out, total_nodes, changed_nodes


def clean_index_dict(records: dict) -> tuple[dict, int, int]:
    cleaned_records = {}
    total_nodes = 0
    changed_nodes = 0
    for key, record in records.items():
        rec = copy.deepcopy(record)
        segs, n_tot, n_chg = clean_segments(rec.get('segments', []))
        rec['segments'] = segs
        cleaned_records[key] = rec
        total_nodes += n_tot
        changed_nodes += n_chg
    return cleaned_records, total_nodes, changed_nodes


def clean_reasoning_root(root: Path) -> dict:
    root = Path(root).resolve()
    combined = {}
    counts = {}
    summary = {}
    for ds in ('r2r', 'rxr'):
        ds_index_path = root / 'generation' / ds / 'cot_index.json'
        if not ds_index_path.is_file():
            continue
        backup_once(ds_index_path)
        source_path = ds_index_path.with_suffix('.uncleaned.json')
        raw_records = json.loads((source_path if source_path.is_file() else ds_index_path).read_text())
        cleaned_records, n_tot, n_chg = clean_index_dict(raw_records)
        atomic_write_json(ds_index_path, cleaned_records)

        cot_new_path = root / 'generation' / ds / 'cot_new.json'
        if cot_new_path.is_file():
            backup_once(cot_new_path)
            native = {}
            for rec in cleaned_records.values():
                eid = str(rec['episode_id'])
                if eid not in native:
                    native[eid] = {'segments': rec['segments']}
            if len(native) == len(cleaned_records):
                atomic_write_json(cot_new_path, native)

        if set(cleaned_records) & set(combined):
            raise ValueError('Duplicate namespaced trajectory key across datasets')
        combined.update(cleaned_records)
        counts[ds] = len(cleaned_records)
        summary[ds] = {'trajectories': len(cleaned_records), 'nodes': n_tot, 'cleaned_nodes': n_chg}

    exports_index = root / 'exports' / 'cot_index.json'
    if combined:
        backup_once(exports_index)
        atomic_write_json(exports_index, combined)
        atomic_write_json(root / 'exports' / 'counts.json', counts)
    elif exports_index.is_file():
        backup_once(exports_index)
        source_path = exports_index.with_suffix('.uncleaned.json')
        raw_records = json.loads((source_path if source_path.is_file() else exports_index).read_text())
        cleaned_records, n_tot, n_chg = clean_index_dict(raw_records)
        atomic_write_json(exports_index, cleaned_records)
        summary['exports'] = {'trajectories': len(cleaned_records), 'nodes': n_tot, 'cleaned_nodes': n_chg}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--root',
        type=Path,
        default=ROOT / 'outputs/reasoning/full_qwen35_27b',
        help='Reasoning output directory containing generation/ and exports/',
    )
    parser.add_argument('--cot-index', type=Path, help='Clean a standalone cot_index.json file instead of --root')
    parser.add_argument('--output', type=Path, help='Output path when --cot-index is specified (defaults to in-place with .uncleaned.json backup)')
    args = parser.parse_args()

    if args.cot_index is not None:
        out_path = args.output or args.cot_index
        if out_path.resolve() == args.cot_index.resolve():
            backup_once(args.cot_index)
            src = args.cot_index.with_suffix('.uncleaned' + args.cot_index.suffix)
            raw = json.loads((src if src.is_file() else args.cot_index).read_text())
        else:
            raw = json.loads(args.cot_index.read_text())
        cleaned, n_tot, n_chg = clean_index_dict(raw)
        atomic_write_json(out_path, cleaned)
        print(f"Cleaned standalone index -> {out_path}: nodes={n_tot}, cleaned_nodes={n_chg}")
        return

    summary = clean_reasoning_root(args.root)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

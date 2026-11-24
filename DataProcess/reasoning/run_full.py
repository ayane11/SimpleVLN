#!/usr/bin/env python3
"""Docker entry point: prepare R2R/RxR, serve one TP=2 teacher, generate and export."""
import argparse
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from prepare_full import save

HERE = Path(__file__).resolve().parent
CHILDREN = []


def start(command, log, env):
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    CHILDREN.append(process)
    return process


def stop(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def interrupted(signum, frame):
    raise KeyboardInterrupt


def export(root):
    combined, counts = {}, {}
    for ds in ('r2r', 'rxr'):
        index = root/'generation'/ds/'cot_index.json'
        records = json.loads(index.read_text()) if index.exists() else {}
        if set(records) & set(combined):
            raise ValueError('Duplicate namespaced trajectory key')
        combined.update(records)
        counts[ds] = len(records)
    save(root/'exports/cot_index.json', combined)
    save(root/'exports/counts.json', counts)
    print('Exported', counts, flush=True)


def run(args):
    root = Path(args.output).resolve()
    (root/'logs').mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus, PYTHONUNBUFFERED='1')
    if len(args.gpus.split(',')) != 2:
        raise ValueError('--gpus requires two GPU IDs, e.g. 2,3')
    for executable in (args.habitat_python, args.teacher_python):
        if not Path(executable).is_file():
            raise FileNotFoundError(executable)
    def execute(command, name):
        print('Running', name, 'log:', root/'logs'/f'{name}.log', flush=True)
        with (root/'logs'/f'{name}.log').open('a') as log:
            return start(command, log, env).wait()
    if args.stage in ('all', 'prepare'):
        for ds in ('r2r', 'rxr'):
            code = execute([args.habitat_python, str(HERE/'prepare_full.py'), '--dataset', ds,
                            '--data-root', args.data_root, '--output', str(root/'preparation'/ds)], f'prepare_{ds}')
            if code:
                raise RuntimeError(f'{ds} preparation failed (exit {code}); rerun the same command to resume')
    if args.stage in ('all', 'generate'):
        for ds in ('r2r', 'rxr'):
            for filename in ('annotations.ready.json', 'events.json', 'status.json'):
                if not (root/'preparation'/ds/filename).is_file():
                    raise FileNotFoundError(f'Run prepare first: {ds}/{filename}')
        if not (Path(args.model)/'config.json').is_file():
            raise FileNotFoundError(args.model)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', args.port))
        command = [args.teacher_python, '-m', 'vllm.entrypoints.openai.api_server', '--model', args.model,
                   '--served-model-name', 'qwen35-27b', '--tensor-parallel-size', '2',
                   '--gpu-memory-utilization', '.85', '--max-model-len', '32768', '--max-num-seqs', '8',
                   '--max-num-batched-tokens', '8192', '--limit-mm-per-prompt', '{"image":16,"video":0}',
                   '--mm-processor-kwargs', '{"max_pixels":262144}', '--skip-mm-profiling',
                   '--enable-prefix-caching', '--mm-encoder-tp-mode', 'data', '--host', '127.0.0.1', '--port', str(args.port)]
        errors = []
        with (root/'logs/teacher.log').open('a') as log:
            server = start(command, log, env)
            try:
                deadline = time.monotonic()+1800
                while True:
                    if server.poll() is not None:
                        raise RuntimeError('Teacher failed to start; see logs/teacher.log')
                    try:
                        with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/health', timeout=3):
                            break
                    except OSError:
                        if time.monotonic() > deadline:
                            raise TimeoutError('Teacher startup timeout; see logs/teacher.log')
                        time.sleep(2)
                for ds in ('r2r', 'rxr'):
                    command = [args.teacher_python, str(HERE/('repair_failed.py' if args.repair_failed else 'generate.py')), '--dataset', ds,
                               '--annotations', str(root/'preparation'/ds/'annotations.ready.json'),
                               '--image-root', str(Path(args.data_root)/'images'/ds),
                               '--events-json', str(root/'preparation'/ds/'events.json'),
                               '--output-dir', str(root/'generation'/ds), '--model', args.model,
                               '--backend', 'vllm', '--base-url', f'http://127.0.0.1:{args.port}/v1',
                               '--workers', str(args.workers), '--expert-progress-v3', '--history-frames', '8',
                               '--max-new-tokens', '3072', '--max-context', '32768', '--retries', '2',
                               '--retry-failed', '--exclude-nodes', str(HERE/'excluded_nodes.json')]
                    seed = root/'seed_v3'/ds
                    if seed.exists():
                        command += ['--reuse-requests-from', str(seed)]
                    if execute(command, f'generate_{ds}'):
                        errors.append(ds)
            finally:
                stop(server)
        export(root)
        if errors:
            raise RuntimeError(f'Incomplete generation: {errors}; successful results saved, rerun to retry failed nodes')
    if args.stage == 'export':
        export(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['all', 'prepare', 'generate', 'export'])
    parser.add_argument('--output', default=str(HERE.parents[1]/'outputs/reasoning/full_qwen35_27b'))
    parser.add_argument('--data-root', default='/data/vln')
    parser.add_argument('--model', default='/workspace/VLN/models/Qwen3.5-27B')
    parser.add_argument('--gpus', default='2,3')
    parser.add_argument('--port', type=int, default=22992)
    parser.add_argument('--repair-failed', action='store_true', help='Use tagged evidence and word boundaries for uncached requests during resume')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--habitat-python', default='/root/miniforge3/envs/simplevln-eval/bin/python')
    parser.add_argument('--teacher-python', default='/root/miniforge3/envs/simplevln-server/bin/python')
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        with (root/'.run.lock').open('a+') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(f'该输出目录已有任务正在运行，本次重复启动已取消。原任务未停止。\n'
                      f'进度请查看：{root / "logs"} 下的 prepare_*.log / generate_*.log',
                      file=sys.stderr, flush=True)
                raise SystemExit(2)
            lock.seek(0)
            lock.truncate()
            lock.write(str(os.getpid()) + '\n')
            lock.flush()
            run(args)
    finally:
        for process in reversed(CHILDREN):
            stop(process)


if __name__ == '__main__':
    main()

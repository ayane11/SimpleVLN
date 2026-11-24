"""Check navigation requests and action execution without Habitat or model weights."""

import argparse
import ast
import base64
from contextlib import nullcontext, redirect_stderr
from copy import deepcopy
from io import BytesIO, StringIO
import os
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from navigation_memory import memory_prompt, parse_reply

TREE = ast.parse((ROOT / 'vln/eval.py').read_text())


class Frame:
    def __init__(self, label):
        self.label = str(label)

    def astype(self, _):
        return self

    def convert(self, _):
        return self

    def save(self, stream, format):
        stream.write(self.label.encode())


class NavigationTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.requests = []
        self.output = 'forward 25 cm'

        def client(**settings):
            def create(**request):
                self.requests.append((settings['base_url'], deepcopy(request)))
                return SimpleNamespace(choices=[SimpleNamespace(
                    message=SimpleNamespace(content=next(self.output) if not isinstance(self.output, str) else self.output))])
            return SimpleNamespace(chat=SimpleNamespace(
                completions=SimpleNamespace(create=create)))

        nodes = [n for n in TREE.body if isinstance(n, (ast.Assign, ast.ClassDef))
                 or isinstance(n, ast.FunctionDef) and n.name == 'encode_image_base64']
        nodes = [n for n in nodes if not isinstance(n, ast.Assign)
                 or any(isinstance(t, ast.Name) and t.id == 'SYSTEM_PROMPT' for t in n.targets)]
        namespace = dict(memory_prompt=memory_prompt, parse_reply=parse_reply, Agent=object, OpenAI=client, os=os, re=re, BytesIO=BytesIO,
                         base64=base64, Image=SimpleNamespace(fromarray=lambda x: x),
                         random=SimpleNamespace(randint=lambda a, b: 3))
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<navigation>', 'exec'), namespace)
        self.agent = namespace['SimpleVLN_Agent'](
            'http://server-a/v1,http://server-b/v1', 'model', self.folder.name,
            25, 15, 10, 1.0)

    def act(self, frame):
        return self.agent.act({'rgb': Frame(frame), 'instruction': {'text': 'Go to the door.'}},
                              {}, 'episode')['action']

    def image_labels(self, request):
        content = request['messages'][1]['content']
        return [base64.b64decode(item['image_url']['url'].split(',', 1)[1]).decode()
                for item in content if item['type'] == 'image_url']

    def test_prompt_images_sampling_and_request_parameters(self):
        for i in range(12):
            self.assertEqual(self.act(i), 1)
        first = self.requests[0][1]
        self.assertEqual(first['messages'][0], {'role': 'system', 'content': [
            {'type': 'text', 'text': 'You are a visual language navigation model.'}]})
        self.assertEqual(self.image_labels(first), ['0', '0'])
        texts = [item['text'] for item in first['messages'][1]['content']
                 if item['type'] == 'text']
        self.assertEqual(texts, [
            'Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations',
            ' and an image of the current observation',
            ". Your assigned task is: 'Go to the door.'. Analyze this series of images to decide your next move, "
            'which could involve turning left or right by a specific degree or moving forward a certain distance.',
        ])
        self.assertEqual((first['model'], first['temperature'], first['top_p'],
                          first['max_completion_tokens']), ('model', .3, .95, 512))
        self.assertEqual([endpoint for endpoint, _ in self.requests[:4]],
                         ['http://server-a/v1', 'http://server-b/v1'] * 2)
        self.assertEqual(self.image_labels(self.requests[-1][1]),
                         ['2', '3', '4', '5', '7', '8', '9', '10', '11'])

    def test_full_prompt_matches_training_converter(self):
        source = ROOT.parent / 'DataProcess/tool/convert_vln.py'
        if not source.exists():
            source = ROOT / 'tests/fixtures/navigation_prompt.json'
            import json
            reference = json.loads(source.read_text(encoding='utf-8'))
            expected_system = reference['system']
            def build_prompt(instruction, count):
                return reference['template'].format('<image>' * (count - 1), '<image>', instruction)
        else:
            tree = ast.parse(source.read_text(encoding='utf-8'))
            nodes = [n for n in tree.body if (
                isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
                    {'SYSTEM_PROMPT', 'VLN_PROMPT_TEMPLATE'} for t in n.targets)) or (
                isinstance(n, ast.FunctionDef) and n.name == 'build_vln_prompt')]
            ns = {}
            exec(compile(ast.Module(body=nodes, type_ignores=[]), '<training-prompt>', 'exec'), ns)
            expected_system, build_prompt = ns['SYSTEM_PROMPT'], ns['build_vln_prompt']
        for frame in range(12):
            self.act(frame)
            request = self.requests[-1][1]
            system = ''.join(x['text'] for x in request['messages'][0]['content'])
            parts = request['messages'][1]['content']
            prompt = ''.join(x['text'] if x['type'] == 'text' else '<image>' for x in parts)
            count = sum(x['type'] == 'image_url' for x in parts)
            self.assertEqual(system, expected_system)
            self.assertEqual(prompt, build_prompt('Go to the door.', count))

    def test_actions_are_limited_and_pending_actions_avoid_extra_requests(self):
        self.output = '<think>plan\nnext move</think>forward 150 cm, left 30 degrees, right 90 degrees'
        self.assertEqual([self.act(i) for i in range(5)], [1, 1, 1, 2, 2])
        self.assertEqual(len(self.requests), 1)
        self.agent.reset()
        self.output = 'stop'
        self.assertEqual(self.act(5), 0)
        self.assertEqual(self.image_labels(self.requests[-1][1]), ['5', '5'])

    def test_two_round_memory_and_action_queue(self):
        self.agent.reasoning = True
        mem = 'I see a hall. The next plan is to move forward.'
        self.output = iter([f'REASON: {mem}', 'ACTION: forward 50 cm', 'ACTION: stop'])
        self.assertEqual([self.act(i) for i in range(3)], [1, 1, 0])
        requests = [request for _, request in self.requests]
        self.assertEqual([self.image_labels(r) for r in requests], [['0', '0'], ['0', '0'], ['0', '1', '2']])
        for r in requests:
            self.assertEqual(r['extra_body'], {'chat_template_kwargs': {'enable_thinking': False}})
            content = r['messages'][1]['content']
            self.assertEqual([item['type'] for item in content[-2:]], ['image_url', 'text'])
        prompts = [r['messages'][1]['content'][-1]['text'] for r in requests]
        base_tail = (
            ". Your assigned task is: 'Go to the door.'. Analyze this series of images to decide your next move, "
            "which could involve turning left or right by a specific degree or moving forward a certain distance."
        )
        self.assertEqual(prompts, [
            base_tail + memory_prompt(''),
            base_tail + memory_prompt(mem, True, steps_ago=0),
            base_tail + memory_prompt(mem, False, steps_ago=2),
        ])
        self.assertIn('The next plan is to move forward.', prompts[1])
        self.assertNotIn('The next plan is to move forward.', prompts[2])
        self.assertEqual((self.agent.decisions, self.agent.reasoning_updates), (2, 1))
        self.agent.reset()
        self.assertEqual((self.agent.memory, self.agent.step_idx, self.agent.last_reason_step), ('', -1, None))

    def test_step_idx_survives_history_window_truncation(self):
        self.agent.reasoning = True
        replies = ['REASON: At start.', 'ACTION: forward 25 cm'] + ['ACTION: forward 25 cm'] * 14
        self.output = iter(replies)
        for i in range(15):
            self.assertEqual(self.act(i), 1)
        self.assertEqual(len(self.agent.rgb_list), 10)
        last_prompt = self.requests[-1][1]['messages'][1]['content'][-1]['text']
        self.assertIn('Navigation memory (updated 14 steps ago): At start.', last_prompt)

    def test_second_reason_cannot_loop_or_execute_stop_from_prose(self):
        self.agent.reasoning = True
        self.output = iter(['REASON: Look around.', 'REASON: Do not stop yet.'])
        self.assertEqual(self.act(0), 3)  # The mocked legacy fallback.
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(self.agent.invalid_mode_replies, 1)

    def test_action_parsing_and_invalid_output_fallback(self):
        for text, expected in [('forward', (1, 25)), ('turn left', (2, 15)),
                               ('turn right', (3, 15)), ('<answer>STOP</answer>', (0, None)),
                               ('unrecognized', (None, None))]:
            with self.subTest(text=text):
                self.assertEqual(self.agent.extract_result(text), expected)
        self.output = 'unrecognized'
        self.assertEqual(self.act(0), 3)


class CommandLineTests(unittest.TestCase):
    def test_video_alias_and_legacy_generation_flag(self):
        node = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        calls = []
        namespace = dict(argparse=argparse, os=__import__("os"), seed_all=lambda: None,
                         resolve_vllm_urls=lambda _: 'http://server/v1',
                         result_directory_lock=lambda _: nullcontext(),
                         run_evaluation=lambda args, _, **kwargs: calls.append(args), print=lambda *a: None)
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<cli>', 'exec'), namespace)
        argv = ['eval.py', '--exp-config', 'config.yaml', '--split-num', '1', '--result-path', 'results']
        for video_flag in ('--save-video', '--save_vedio'):
            with patch('sys.argv', argv + [video_flag, '--num-generations', '1']), \
                 patch.dict('sys.modules', {'wandb': MagicMock()}), \
                 patch('pathlib.Path.mkdir'):
                namespace['main']()
            self.assertTrue(calls[-1].save_video)
        with patch('sys.argv', argv + ['--num-generations', '2']), redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit) as error:
                namespace['main']()
        self.assertEqual(error.exception.code, 2)


if __name__ == '__main__':
    unittest.main()

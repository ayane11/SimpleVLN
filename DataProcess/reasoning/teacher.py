"""Local Transformers model or an explicitly configured local vLLM endpoint."""
import base64
import copy
import json
import time
import urllib.request
import urllib.error
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit


def processor_messages(messages):
    """Transformers multimodal processors expect typed content even for system/assistant turns."""
    result = copy.deepcopy(messages)
    for message in result:
        if isinstance(message['content'], str):
            message['content'] = [{'type': 'text', 'text': message['content']}]
    return result


def check_weights(model_path):
    path = Path(model_path)
    if not (path / "config.json").is_file():
        raise ValueError(f"Missing local model config: {path}")
    index = path / "model.safetensors.index.json"
    files = set(json.loads(index.read_text())["weight_map"].values()) if index.exists() else {"model.safetensors"}
    missing = sorted(n for n in files if not (path / n).is_file() or (path / n).stat().st_size == 0)
    if missing:
        raise ValueError(f"Model download incomplete: {missing}. Resume hf download first.")


class TransformersTeacher:
    def __init__(self, args):
        check_weights(args.model)
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.torch, self.args = torch, args
        if not torch.cuda.is_available():
            raise ValueError("Local BF16 teacher requires CUDA GPUs; use --dry-run for CPU validation")
        self.processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
        # Preserve patch/normalization parameters; cap resolution to bound teacher memory.
        self.processor.image_processor.size = dict(shortest_edge=65536, longest_edge=args.max_pixels)
        self.model = AutoModelForImageTextToText.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="auto", local_files_only=True,
            max_memory={**{i: f"{args.gpu_memory_gib}GiB" for i in range(torch.cuda.device_count())},
                        "cpu": "0GiB"},
            attn_implementation="sdpa",
        ).eval()

    def generate(self, messages):
        t = time.monotonic()
        inputs = self.processor.apply_chat_template(
            processor_messages(messages), tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors="pt", enable_thinking=False,
        )
        n = inputs["input_ids"].shape[-1]
        if n + self.args.max_new_tokens > self.args.max_context:
            raise ValueError(f"Prompt {n} + output budget exceeds max_context; reduce global_frames/max_pixels")
        inputs = inputs.to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.args.max_new_tokens, do_sample=False)
        generated = output[0, n:]
        text = self.processor.decode(generated, skip_special_tokens=True)
        if len(generated) >= self.args.max_new_tokens:
            raise ValueError("Output reached max_new_tokens; increase budget (do not accept truncated JSON)")
        return dict(text=text, input_tokens=n, output_tokens=len(generated), seconds=time.monotonic()-t)


class VLLMTeacher:
    def __init__(self, args):
        self.args = args
        url = urlsplit(args.base_url)
        if url.scheme not in ("http", "https") or url.hostname not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError("Only local vLLM endpoints are supported; no cloud API calls")
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def generate(self, messages):
        started = time.monotonic()
        messages = copy.deepcopy(messages)
        for message in messages:
            if not isinstance(message["content"], list):
                continue
            for item in message["content"]:
                if item["type"] == "image":
                    # Bound uploaded resolution as well as server preprocessing.
                    path = Path(item.pop('image'))
                    stat = path.stat()
                    uri = encoded_image(str(path), stat.st_mtime_ns, stat.st_size, self.args.max_pixels)
                    item.update(type="image_url", image_url={"url": uri})
        body = dict(model=self.args.served_model, messages=messages,
                    temperature=getattr(self.args, 'temperature', 0),
                    max_tokens=self.args.max_new_tokens, chat_template_kwargs={"enable_thinking": False})
        if hasattr(self.args, 'top_p'):
            body['top_p'] = self.args.top_p
        if hasattr(self.args, 'generation_seed'):
            body['seed'] = self.args.generation_seed
        req = urllib.request.Request(self.args.base_url.rstrip('/') + '/chat/completions',
                                     data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        t = time.monotonic()
        try:
            with self.opener.open(req, timeout=self.args.timeout) as response:
                value = json.load(response)
        except urllib.error.HTTPError as exc:
            raise ValueError(f'vLLM HTTP {exc.code}: {exc.read().decode()[:2000]}') from exc
        choice = value['choices'][0]
        if choice['finish_reason'] != 'stop' or not choice['message'].get('content'):
            raise ValueError(f"Incomplete generation: {choice['finish_reason']}")
        usage = value.get('usage', {})
        return dict(text=choice['message']['content'], input_tokens=usage.get('prompt_tokens'),
                    output_tokens=usage.get('completion_tokens'), seconds=time.monotonic()-started,
                    http_seconds=time.monotonic()-t, endpoint=self.args.base_url)


@lru_cache(maxsize=512)
def encoded_image(path, mtime_ns, size, max_pixels):
    """Reuse full-trajectory images between global and node turns; bounded CPU cache."""
    from PIL import Image
    from io import BytesIO
    with Image.open(path) as image:
        image = image.convert('RGB')
        scale = min(1.0, (max_pixels / (image.width * image.height)) ** .5)
        if scale < 1:
            image = image.resize((max(1, int(image.width*scale)), max(1, int(image.height*scale))))
        buf = BytesIO(); image.save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()

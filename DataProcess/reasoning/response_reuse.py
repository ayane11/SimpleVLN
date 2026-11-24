"""Reuse exact local model requests after validator-only changes, with provenance."""
import copy,json
from pathlib import Path
from core import digest

PROFILE=('model','backend','served_model','max_pixels','max_new_tokens','max_context',
         'thinking','decoding','dtype','model_files','weight_files','library_versions')

def profile(settings):
    return {k:settings.get(k) for k in PROFILE}


def request_identity(messages, images, settings):
    return digest(dict(messages=messages,images=images,generation=profile(settings)))


def load_responses(directory, settings):
    directory=Path(directory);out={};image_snapshot={}
    for p in (directory/'episodes').glob('*.json'):
        for image in json.loads(p.read_text()).get('image_files',[]):image_snapshot[image['path']]=image
    for folder in ['requests','errors']:
        for p in sorted((directory/folder).glob('*.json')):
            value=json.loads(p.read_text())
            if profile(value.get('settings',{}))!=profile(settings):continue
            messages=value.get('messages')
            if not messages:continue
            paths=[c['image'] for m in messages if isinstance(m['content'],list) for c in m['content'] if c['type']=='image']
            if folder=='requests':
                candidates=[(messages,value.get('response'),value.get('image_files',[]))]
            else:
                # Error logs store the messages for their LAST attempt, including
                # repair feedback. Never pair these with an earlier response.
                if not value.get('attempts') or any(path not in image_snapshot for path in paths):continue
                candidates=[(messages,value['attempts'][-1].get('response'),[image_snapshot[path] for path in paths])]
            for messages,response,images in candidates:
                if not response or not response.get('text'):continue
                key=request_identity(messages,images,settings)
                reused=copy.deepcopy(response);reused['reused_from']=str(p.resolve())
                reused['original_seconds']=reused.get('seconds');reused['seconds']=0
                out.setdefault(key,reused)
    return out

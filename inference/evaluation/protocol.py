"""Prompt construction and configuration without model dependencies."""
from pathlib import Path
import json

REPO_ROOT = Path(__file__).resolve().parents[2]


def prompt_text(name):
    return (REPO_ROOT / 'prompts' / name).read_text(encoding='utf-8').strip()


def load_labels(path):
    labels = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(labels, list) or not labels:
        raise ValueError('Labels must be a nonempty JSON array.')
    if any(not isinstance(x, str) or not x.strip() for x in labels):
        raise ValueError('Each label must be a nonempty string.')
    if len(set(labels)) != len(labels):
        raise ValueError('Candidate labels must be unique.')
    return labels


def build_messages(image_path, mode='ddi', labels=None):
    """Build a diagnostic request without any reference diagnosis field."""
    if mode == 'ddi':
        if labels is not None:
            raise ValueError('DDI open-ended inference does not accept candidate labels.')
        messages = [{'role': 'system', 'content': prompt_text('ddi_system.txt')}]
        user_text = prompt_text('ddi_user.txt')
    elif mode == 'classification':
        if not labels:
            raise ValueError('Classification requires candidate labels.')
        messages = []
        user_text = prompt_text('classification_user.txt').replace(
            '{candidate_labels}', '\n'.join(labels)
        )
    else:
        raise ValueError(f'Unknown inference mode: {mode}')
    messages.append({'role': 'user', 'content': [
        {'type': 'image', 'image': str(image_path)},
        {'type': 'text', 'text': user_text},
    ]})
    return messages


def generation_kwargs(config):
    result = {
        'max_new_tokens': config['max_new_tokens'],
        'do_sample': config['do_sample'],
        'repetition_penalty': config['repetition_penalty'],
        'use_cache': config['use_cache'],
        'num_beams': 1,
        'num_return_sequences': 1,
        'no_repeat_ngram_size': 0,
    }
    if config['do_sample']:
        result.update(temperature=config['temperature'], top_p=config['top_p'])
    return result

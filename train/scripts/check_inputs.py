"""Check external assets required by the SFT launcher, without loading a model."""
import json
import os
from pathlib import Path

root = Path(__file__).resolve().parents[1]
model = Path(os.environ.get('SKINGPT_MODEL_DIR', root / 'external/base_model'))
dataset = Path(os.environ.get('SKINGPT_DATASET_DIR', root / 'data'))
images = Path(os.environ.get('SKINGPT_IMAGE_ROOT', root / 'external/images'))
features = Path(os.environ.get('SKINGPT_FEATURE_ROOT', root / 'external/teacher_features'))
errors = []
for label, path in [('base model config', model / 'config.json'),
                    ('dataset registry', dataset / 'dataset_info.json')]:
    if not path.is_file():
        errors.append(f'Missing {label}: {path}')
for label, path in [('images', images), ('teacher features', features)]:
    if not path.is_dir():
        errors.append(f'Missing {label} directory: {path}')
if (dataset / 'dataset_info.json').is_file():
    entries = json.loads((dataset / 'dataset_info.json').read_text())
    expected = json.loads((root / 'data/dataset_info.json').read_text())
    for name, registration in expected.items():
        path = dataset / registration['file_name']
        if not path.is_file():
            errors.append(f'Missing training file: {path}')
        if entries.get(name) != registration:
            errors.append(f'Dataset registration differs from the registered SFT mapping: {name}')
if errors:
    raise SystemExit('External training assets must be supplied:\n- ' + '\n- '.join(errors))
print('External asset locations are present. This check does not validate every image, feature, or model weight.')

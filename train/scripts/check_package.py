"""Check syntax, imported local modules, file types and public package hashes."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
allowed = {'.py', '.sh', '.json', '.md', '.yaml', '.yml', '.txt', '.cff', '.sha256'}
files = [p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
modules = {}
for path in (root / 'src').rglob('*.py'):
    parts = list(path.relative_to(root / 'src').with_suffix('').parts)
    if parts[-1] == '__init__':
        parts.pop()
    modules['.'.join(parts)] = path
errors = []
for path in files:
    rel = path.relative_to(root).as_posix()
    if path.is_symlink() or path.suffix not in allowed or path.stat().st_size > 2 * 1024 * 1024:
        errors.append(f'Unexpected file: {rel}')
        continue
    text = path.read_text(encoding='utf-8')
    if path.suffix == '.py':
        ast.parse(text, filename=rel)
    if path.suffix == '.json':
        json.loads(text)
for module, path in modules.items():
    package = module if path.name == '__init__.py' else module.rpartition('.')[0]
    for node in ast.walk(ast.parse(path.read_text())):
        targets = []
        if isinstance(node, ast.Import):
            targets = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            targets = [importlib.util.resolve_name('.' * node.level + (node.module or ''), package)
                       if node.level else node.module or '']
        for target in targets:
            if target.startswith('llamafactory') and target not in modules:
                errors.append(f'Missing local import: {module} -> {target}')
manifest = root / 'MANIFEST.sha256'
expected_paths = set()
for line in manifest.read_text().splitlines():
    digest, relative = line.split('  ', 1)
    expected_paths.add(relative)
    path = root / relative
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        errors.append(f'Checksum mismatch: {relative}')
if expected_paths != {p.relative_to(root).as_posix() for p in files if p != manifest}:
    errors.append('Checksum manifest and package file set differ')
if errors:
    raise SystemExit('\n'.join(errors))
print(f'Passed: {len(files)} package files, {len(modules)} source modules, syntax, imports and hashes.')

#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import re
import sys
from pathlib import Path

FORBIDDEN_FILE_PATTERNS = [
    r'\.pdf$',
    r'\.pptx$',
    r'\.key$',
    r'\.ckpt$',
    r'\.safetensors$',
    r'\.pt$',
    r'\.pth$',
    'ref' + 'erences',
    'b' + 'ib',
]

FORBIDDEN_TEXT_PATTERNS = [
    '/Users/' + r'[^\s]+',
    '/home/' + r'[^\s]+',
    r'~/[A-Za-z0-9_./-]+',
    r'github\.com/[A-Za-z0-9_-]+/',
    'acknow' + r'ledg',
    'corresponding' + r'\s+' + 'auth' + 'or',
    'affil' + r'iation',
]

SKIP_DIRS = {'.git', '__pycache__', '.pytest_cache', '.mypy_cache'}
TEXT_SUFFIXES = {'.py', '.md', '.txt', '.yaml', '.yml', '.toml', '.json', '.jsonl', '.sh'}


def main() -> None:
    root = Path('.').resolve()
    hits = []
    for path in root.rglob('*'):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        rel = path.relative_to(root)
        if str(rel) == 'scripts/check_release.py':
            continue
        if not path.is_file():
            continue
        for pat in FORBIDDEN_FILE_PATTERNS:
            if re.search(pat, path.name, re.I):
                hits.append(f'file-name:{rel}')
        if path.suffix.lower() in TEXT_SUFFIXES:
            text = path.read_text(encoding='utf-8', errors='ignore')
            for pat in FORBIDDEN_TEXT_PATTERNS:
                if re.search(pat, text, re.I):
                    hits.append(f'text:{rel}:{pat}')
    if hits:
        print('Potential release leaks found:')
        for h in hits:
            print('  -', h)
        sys.exit(1)
    print('No obvious release leaks found.')


if __name__ == '__main__':
    main()

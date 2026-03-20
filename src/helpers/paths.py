"""
Central configuration for project paths.

Set environment variables to override defaults:
  - PRL_PROJECT_ROOT: project root directory (default: parent of helpers/)
  - PRL_DATA_ROOT: data root directory (default: /media/smbshare/srs-9/prl_project/data)
  - PRL_TRAIN_ROOT: training root directory (default: /media/smbshare/srs-9/prl_project/training)
"""

import os
import json
import re
from pathlib import Path

# Root paths — set env vars to override
PROJECT_ROOT = Path(os.environ.get('PRL_PROJECT_ROOT', Path(__file__).parent.parent))
DATA_ROOT = Path(os.environ.get('PRL_DATA_ROOT', '/media/smbshare/srs-9/prl_project/data'))
TRAIN_ROOT = Path(os.environ.get('PRL_TRAIN_ROOT', '/media/smbshare/srs-9/prl_project/training'))

# Token replacements
TOKEN_MAP = {
    '${PROJECT_ROOT}': str(PROJECT_ROOT),
    '${DATA_ROOT}': str(DATA_ROOT),
    '${TRAIN_ROOT}': str(TRAIN_ROOT),
}


def expand_tokens(value):
    """Recursively expand ${VAR} tokens in strings, lists, and dicts."""
    if isinstance(value, str):
        for token, replacement in TOKEN_MAP.items():
            value = value.replace(token, replacement)
        return value
    elif isinstance(value, list):
        return [expand_tokens(v) for v in value]
    elif isinstance(value, dict):
        return {k: expand_tokens(v) for k, v in value.items()}
    else:
        return value


def strip_json_comments(text):
    """Strip // line comments from JSONC text. Does not strip inside strings."""
    result = []
    for line in text.splitlines():
        # Find // that is not inside a string
        in_string = False
        escape = False
        for i, ch in enumerate(line):
            if escape:
                escape = False
                continue
            if ch == '\\' and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
            if ch == '/' and not in_string and i + 1 < len(line) and line[i + 1] == '/':
                line = line[:i]
                break
        result.append(line)
    return '\n'.join(result)


def load_config(config_path):
    """
    Load a JSON/JSONC config file and expand all ${VAR} tokens.

    Supports .jsonc files with // line comments.
    Paths are resolved relative to os.cwd() if relative.
    """
    config_path = Path(config_path)
    if not config_path.is_absolute():
        curr_dir = Path(os.getcwd())
        config_path = curr_dir / config_path
        print(config_path)
    with open(config_path) as f:
        text = f.read()

    if config_path.suffix == '.jsonc':
        text = strip_json_comments(text)

    config = json.loads(text)
    return expand_tokens(config)

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


def load_config(config_path):
    """
    Load a JSON config file and expand all ${VAR} tokens.

    Paths are resolved relative to PROJECT_ROOT if relative.
    """
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    with open(config_path) as f:
        config = json.load(f)

    return expand_tokens(config)

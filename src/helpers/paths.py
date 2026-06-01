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
from loguru import logger

import yaml

# Root paths — set env vars to override
PROJECT_ROOT = Path(os.environ.get('PRL_PROJECT_ROOT', Path(__file__).parent.parent.parent))
DATA_ROOT = Path(os.environ.get('PRL_DATA_ROOT', '/media/smbshare/srs-9/prl_project/data'))
TRAIN_ROOT = Path(os.environ.get('PRL_TRAIN_ROOT', '/media/smbshare/srs-9/prl_project/training'))


RESOURCES_DIR = PROJECT_ROOT / "src/resources"


def expand_tokens(value, token_map):
    """Recursively expand ${VAR} tokens using a provided mapping dictionary."""
    if isinstance(value, str):
        for token, replacement in token_map.items():
            value = value.replace(token, str(replacement))
        return value
    elif isinstance(value, list):
        return [expand_tokens(v, token_map) for v in value]
    elif isinstance(value, dict):
        return {k: expand_tokens(v, token_map) for k, v in value.items()}
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


def load_config(config_path, pwd=None):
    """
    Load a JSON/JSONC/YAML config file and expand all ${VAR} tokens.

    Supports .jsonc files with // line comments and .yaml/.yml files.
    Paths are resolved relative to pwd (or current working directory if None).
    """
    # 1. Safely resolve config_path relative to pwd or CWD without os.chdir()
    config_path = Path(config_path)
    
    if not config_path.is_absolute():
        base_dir = Path(pwd) if pwd is not None else Path.cwd()
        config_path = (base_dir / config_path).resolve()

    # 2. Read the file
    with open(config_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 3. Parse formats
    if config_path.suffix in ('.yaml', '.yml'):
        config = yaml.safe_load(text)
    else:
        if config_path.suffix == '.jsonc':
            text = strip_json_comments(text)
        config = json.loads(text)

    # 4. Generate the dynamic token map at the exact time of the call
    # If the user passed a specific pwd folder, we use that folder as ${PWD}.
    # Otherwise, we grab the current shell directory at this exact millisecond.
    active_pwd = Path(pwd).resolve() if pwd is not None else Path.cwd()
    
    dynamic_token_map = {
        '${PROJECT_ROOT}': PROJECT_ROOT,
        '${DATA_ROOT}': DATA_ROOT,
        '${TRAIN_ROOT}': TRAIN_ROOT,
        '${PWD}': active_pwd,
    }

    # 5. Expand and return
    return expand_tokens(config, dynamic_token_map)


def load_config0(config_path):
    """
    Load a JSON/JSONC/YAML config file and expand all ${VAR} tokens.

    Supports .jsonc files with // line comments and .yaml/.yml files.
    Paths are resolved relative to os.cwd() if relative.
    """
    config_path = Path(config_path)
    if not config_path.is_absolute():
        curr_dir = Path(os.getcwd())
        config_path = curr_dir / config_path
        print(config_path)
    with open(config_path) as f:
        text = f.read()

    if config_path.suffix in ('.yaml', '.yml'):
        config = yaml.safe_load(text)
    else:
        if config_path.suffix == '.jsonc':
            text = strip_json_comments(text)
        config = json.loads(text)

    return expand_tokens(config)


def get_infer_path(dataset, test_case, experiment_name) -> Path:
    case_dir = dataset.lesion_dir(test_case)
    matches = list(case_dir.glob(f"*{experiment_name.replace('/','_')}.nii.gz"))
    if len(matches) > 1:
        logger.warning(f"Found more than 1 case: {','.join(matches)}, returning the first")
    return matches[0]
    
def find_inference(search_path, experiment_name) -> Path:
    matches = list(search_path.glob(f"*{experiment_name.replace('/','_')}.nii.gz"))
    if len(matches) > 1:
        logger.warning(f"Found more than 1 case: {','.join(matches)}, returning the first")
    return matches[0]


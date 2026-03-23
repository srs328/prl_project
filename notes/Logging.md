# Logging

## Loguru

Loguru uses a single global logger, but its add() accepts a filter callable that receives the full record — including record["name"] (the module). So you can do exactly what you want:

In a notebook cell — run once at the top

```python
import sys
from loguru import logger

# Map module names (or substrings) to minimum log levels.
# Anything not listed gets the default level.
MODULE_LEVELS = {
    "generate_fold_predictions": "WARNING",   # quiet during predict()
    "inference": "INFO",
    "grid": "DEBUG",
}
DEFAULT_LEVEL = "INFO"

def _module_filter(record):
    name = record["name"]  # e.g. "scripts.generate_fold_predictions"
    for module, level in MODULE_LEVELS.items():
        if module in name:
            return record["level"].no >= logger.level(level).no
    return record["level"].no >= logger.level(DEFAULT_LEVEL).no

logger.remove()  # drop the default stderr handler
logger.add(sys.stderr, filter=_module_filter, format="{time:HH:mm:ss} | {level:<7} | {name} - {message}")
```

Then to change the level for a module mid-session, just update the dict and re-add:

```python
MODULE_LEVELS["generate_fold_predictions"] = "DEBUG"
logger.remove()
logger.add(sys.stderr, filter=_module_filter, ...)
```

It's a bit more manual than stdlib's getLogger().setLevel(), but the control is equivalent. The key is that record["name"] is the Python module path (scripts.generate_fold_predictions), so partial string matching works cleanly.

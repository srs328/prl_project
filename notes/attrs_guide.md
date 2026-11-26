# attrs in the PRL Pipeline — A Focused Guide

This guide explains the specific attrs features used in `src/core/configs.py` and *why* each was chosen. It's not a full attrs tutorial — just the pieces that matter here.

---

## What is attrs?

attrs is a library for writing Python classes without boilerplate. Instead of writing `__init__`, `__repr__`, `__eq__`, `__hash__` by hand, you declare fields and attrs generates them.

```python
import attrs

@attrs.define
class Foo:
    x: int
    y: str = "hello"

f = Foo(x=1)
# f.x == 1, f.y == "hello"
# repr: Foo(x=1, y='hello')
# equality: Foo(x=1) == Foo(x=1)  → True
```

The two main decorators are:
- `@attrs.define` — modern, recommended, uses slots by default
- `@attrs.define(frozen=True)` — same but immutable (see below)

---

## `@attrs.define` vs `@attrs.define(frozen=True)`

### `@attrs.define` — mutable

Used on `AlgoConfig` and `SegResNetConfig`.

These configs need to be created, passed around, and occasionally mutated (e.g., evolve a field for a grid sweep). They do **not** need to be hashable or used as dict keys.

```python
@attrs.define
class AlgoConfig:
    learning_rate: float = 0.0002
    num_epochs: int = 500
```

### `@attrs.define(frozen=True)` — immutable + hashable

Used on `PreprocessingConfig`.

Frozen means:
1. **Immutable** — attrs raises `FrozenInstanceError` if you try to set an attribute after construction.
2. **Hashable** — because it's immutable, attrs generates `__hash__`. This lets you use instances as dict keys or set members.

Why does `PreprocessingConfig` need this?

`ExperimentGrid` may generate many experiments where several share identical preprocessing parameters (e.g., you're only sweeping learning rate, not roi expansion). To avoid redundant preprocessing runs, the grid deduplicates by putting configs into a `set`. That requires hashability.

```python
configs = {PreprocessingConfig(expand_xy=20), PreprocessingConfig(expand_xy=20)}
len(configs)  # → 1 (deduplicated)
```

With `@attrs.define` (non-frozen), this would fail with `TypeError: unhashable type`.

---

## `attrs.Factory` — mutable default values

A classic Python gotcha: **never use a mutable object as a default argument**.

```python
# WRONG — all instances share the same list object
class Bad:
    x: list = []

# CORRECT with plain Python — but verbose
class Ok:
    def __init__(self):
        self.x = []
```

attrs handles this cleanly with `attrs.Factory`:

```python
roi_size: list[int] = attrs.Factory(lambda: [44, 44, 8])
extra: dict = attrs.Factory(dict)
```

`Factory` accepts either a callable (called with no args) or a `lambda`. Each new instance gets its own fresh object. This is used for `roi_size`, `extra`, and any other list/dict field with a non-trivial default.

---

## `attrs.field()` — explicit field configuration

When you need more than just a type annotation and default, use `attrs.field()`:

```python
expand_xy: int = attrs.field(default=20, validator=attrs.validators.ge(0))
images: tuple[str, ...] = attrs.field(default=("flair", "phase"), converter=tuple)
```

### `validator`

Runs after `__init__` to assert a constraint. `attrs.validators.ge(0)` means "greater than or equal to 0". If violated, raises `ValueError` at construction time — not silently at runtime when the bad value is used.

```python
PreprocessingConfig(expand_xy=-5)
# → ValueError: ("'expand_xy' must be >= 0 ...")
```

### `converter`

Transforms the value before storing it. Here, `converter=tuple` converts any sequence (list, generator, etc.) passed for `images` into a tuple. This matters for frozen/hashable configs — lists aren't hashable, tuples are.

```python
PreprocessingConfig(images=["flair", "phase"])  # list input
# stored as: ("flair", "phase")                 # tuple
```

---

## `attrs.evolve()` — creating modified copies

Because frozen instances can't be mutated, attrs provides `evolve()` to create a new instance with specific fields changed:

```python
base = PreprocessingConfig(expand_xy=20, expand_z=2)
wider = attrs.evolve(base, expand_xy=40)
# base is unchanged; wider has expand_xy=40, expand_z=2
```

`evolve()` also works on mutable configs — it's just a convenient copy-with-overrides. The grid uses this to generate the Cartesian product of parameter combinations without touching the base config.

---

## `attrs.fields()` and `attrs.fields_dict()` — introspection

These let you iterate over a class's fields at runtime, without hardcoding field names. The pipeline uses this heavily to avoid manual enumeration.

### `attrs.fields(cls)` → tuple of `Attribute` objects

```python
for field in attrs.fields(AlgoConfig):
    print(field.name, getattr(some_instance, field.name))
```

This is how `to_input_dict()` builds the AutoRunner dict dynamically:

```python
for field in attrs.fields(type(self)):
    if field.name in self._SKIP_IN_INPUT:
        continue
    val = getattr(self, field.name)
    if val is not None:
        d[field.name] = val
```

Adding a new field to `AlgoConfig` automatically appears in the output dict — no manual update needed.

### `attrs.fields_dict(cls)` → `{name: Attribute}` mapping

Used in `from_dict()` to check which keys in an incoming dict are known fields:

```python
known = attrs.fields_dict(config_cls)
known_params = {k: v for k, v in d.items() if k in known}
extra_params  = {k: v for k, v in d.items() if k not in known}
```

This is how `from_dict()` separates recognized fields from unknown ones (which land in `extra`), without enumerating every field name explicitly.

---

## `ClassVar` — class-level metadata, not instance fields

attrs only treats annotations as fields if they're instance-level. `ClassVar[...]` (from `typing`) signals "this is a class variable, not an instance field" — attrs skips it.

```python
from typing import ClassVar

_SKIP_IN_INPUT: ClassVar[frozenset] = frozenset({"algo", "extra"})
_NETWORK_KEY_MAP: ClassVar[dict[str, str]] = {"init_filters": "network#init_filters", ...}
```

These are class-level constants used by methods — they don't vary per-instance and shouldn't appear in `__repr__`, `__eq__`, or `to_input_dict()`. Without `ClassVar`, attrs would try to make them constructor arguments.

---

## Inheritance with `@attrs.define`

`SegResNetConfig` subclasses `AlgoConfig` and adds network-architecture fields:

```python
@attrs.define
class SegResNetConfig(AlgoConfig):
    algo: str = "segresnet"
    init_filters: int | None = None
    blocks_down: list[int] | None = None
    ...
```

attrs handles inheritance correctly — `SegResNetConfig` gets all of `AlgoConfig`'s fields plus its own. `attrs.fields(SegResNetConfig)` returns all of them together.

The subclass also overrides `to_input_dict()` to remap the network fields to MONAI's `network#key` nested syntax, while calling `super()` to handle the shared fields:

```python
def to_input_dict(self, datalist_path, dataroot) -> dict:
    d = super().to_input_dict(datalist_path, dataroot)
    for attr_name, monai_key in self._NETWORK_KEY_MAP.items():
        if attr_name in d:
            d[monai_key] = d.pop(attr_name)
    return d
```

---

## The Registry Pattern

A plain dict maps algo name strings to config subclasses:

```python
_ALGO_REGISTRY: dict[str, type[AlgoConfig]] = {}
_ALGO_REGISTRY["segresnet"] = SegResNetConfig
```

`AlgoConfig.from_dict()` uses this to dispatch to the right subclass when loading from JSON/YAML:

```python
config_cls = _ALGO_REGISTRY.get(algo, cls)  # fallback to AlgoConfig
```

This is intentionally simple. Adding support for a new algo (e.g., `swinunetr`) means:
1. Write `SwinUNETRConfig(AlgoConfig)` with its specific fields
2. Add `_ALGO_REGISTRY["swinunetr"] = SwinUNETRConfig`

No changes to `from_dict()` or any calling code.

---

## Summary Table

| Feature | Where used | Why |
|---|---|---|
| `@attrs.define` | `AlgoConfig`, `SegResNetConfig` | Auto-generates `__init__`, `__repr__`, `__eq__` |
| `@attrs.define(frozen=True)` | `PreprocessingConfig` | Immutable + hashable; enables set-based deduplication in grid |
| `attrs.Factory(lambda: [...])` | `roi_size`, `extra` | Safe mutable defaults (avoids shared-object bug) |
| `attrs.field(validator=...)` | `expand_xy`, `expand_z` | Fail fast on bad values at construction |
| `attrs.field(converter=tuple)` | `images` | Normalize input type; ensure hashability |
| `attrs.evolve()` | Grid sweeps, config variants | Copy-with-overrides for immutable or mutable configs |
| `attrs.fields(cls)` | `to_input_dict()`, `to_monai_config_dict()` | Dynamic field iteration; new fields flow through automatically |
| `attrs.fields_dict(cls)` | `from_dict()` | Separate known vs unknown keys without enumerating names |
| `ClassVar` | `_SKIP_IN_INPUT`, `_NETWORK_KEY_MAP` | Class constants; excluded from attrs field machinery |
| Inheritance | `SegResNetConfig(AlgoConfig)` | Shared base fields + algo-specific overrides |
| Registry dict | `_ALGO_REGISTRY` | String → subclass dispatch in `from_dict()` |

"""Configuration and storage shared by the browser learner and its policy."""
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

QWEN_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
QWEN_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"


@contextmanager
def _atomic_path(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as f:
        temp = Path(f.name)
    try:
        yield temp
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def reservoir_slot(rng, seen, size, capacity):
    """Return the insertion/replacement index for the latest distinct arrival."""
    slot = size if size < capacity else rng.randrange(seen)
    return slot if slot < capacity else None

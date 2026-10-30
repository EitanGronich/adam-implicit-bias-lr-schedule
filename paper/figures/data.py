"""Read and update the numerical sections of an experiment archive."""
from pathlib import Path

import numpy as np


FORMAT = "adam_paper_data_v1"


def load_section(path, section, *, required=True):
    """Return one figure-data section without exposing its storage prefix."""
    prefix = section + "__"
    with np.load(path, allow_pickle=False) as archive:
        if "format" not in archive or str(archive["format"].item()) != FORMAT:
            raise ValueError("Unsupported experiment archive format")
        values = {key[len(prefix):]: archive[key].copy()
                  for key in archive.files if key.startswith(prefix)}
    if not values and required:
        raise ValueError(f"Experiment archive has no {section!r} section")
    return values


def write_sections(path, **sections):
    """Replace named sections atomically, preserving other experiment data."""
    path = Path(path)
    arrays = {"format": np.asarray(FORMAT)}
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            if "format" not in archive or str(archive["format"].item()) != FORMAT:
                raise ValueError("Unsupported experiment archive format")
            arrays.update({key: archive[key].copy() for key in archive.files})
    for name, values in sections.items():
        if not name or "__" in name or not values:
            raise ValueError("Sections require a nonempty name and array mapping")
        prefix = name + "__"
        arrays = {key: value for key, value in arrays.items() if not key.startswith(prefix)}
        for key, value in values.items():
            if not key or "__" in key:
                raise ValueError("Array names must be nonempty and cannot contain '__'")
            value = np.asarray(value)
            if value.dtype.hasobject:
                raise ValueError("Experiment archives cannot contain object arrays")
            arrays[prefix + key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)

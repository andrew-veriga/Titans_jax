"""
HuggingFace Hub checkpoint utilities for Titans training.

Save/load model checkpoints (zip archives) with hyperparameter metadata
to/from a HuggingFace dataset repository.

Usage in notebooks:
    from hf_checkpoint import save_checkpoint_to_hf, load_checkpoint_from_hf
"""

import json
import os
import shutil
import tempfile
from typing import Any

from huggingface_hub import HfApi, hf_hub_download, login


# ── Helpers ──────────────────────────────────────────────────────────────

# ── Schedule serialization (Variant 1) ──────────────────────────────────
# optax schedules are closures (plain functions), so they cannot be
# introspected at runtime.  Instead, notebooks store schedule **parameters**
# as dicts with a ``"_schedule"`` key:
#
#     opt_params = {
#         "lr_muon": {"_schedule": "warmup_cosine_decay",
#                     "init_value": 5e-4, "peak_value": 5e-4, ...},
#         "beta": 0.90,
#         ...
#     }
#
# ``reconstruct_schedule()`` / ``reconstruct_opt_params()`` convert these
# dicts back into callable optax schedules.


def schedule(name: str, **kwargs) -> dict[str, Any]:
    """Create a serializable schedule descriptor.

    Usage in notebooks::

        opt_params = {
            "lr_muon": schedule("warmup_cosine_decay",
                                init_value=5e-4, peak_value=5e-4,
                                warmup_steps=1500, decay_steps=28500,
                                end_value=1e-5),
            "adam_b1": schedule("linear",
                                init_value=0.7, end_value=0.90,
                                transition_steps=2000, transition_begin=1500),
            "beta": 0.90,
        }

    This dict is JSON-serializable out of the box and can be passed directly
    to ``save_checkpoint_to_hf(opt_params=...)``.

    To get back a callable schedule, use ``reconstruct_schedule()`` or
    ``reconstruct_opt_params()``.
    """
    return {"_schedule": name, **kwargs}


class _ScheduleEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy/JAX scalars and falls back to repr."""

    def default(self, obj: Any) -> Any:
        if hasattr(obj, "item"):
            return obj.item()
        try:
            return repr(obj)
        except Exception:
            return super().default(obj)


def reconstruct_schedule(config: dict[str, Any]):
    """Reconstruct an optax schedule from a serialized dict.

    Args:
        config: Dict with ``"_schedule"`` key and schedule parameters.

    Returns:
        A callable optax schedule.
    """
    import optax

    type_name = config.get("_schedule")
    if type_name is None:
        raise ValueError("Dict is not a serialized schedule (missing '_schedule' key)")

    constructors = {
        "warmup_cosine_decay": optax.warmup_cosine_decay_schedule,
        "cosine_decay": optax.cosine_decay_schedule,
        "linear": optax.linear_schedule,
        "constant": optax.constant_schedule,
        "polynomial": optax.polynomial_schedule,
    }

    ctor = constructors.get(type_name)
    if ctor is None:
        raise ValueError(f"Unknown schedule type: {type_name}")

    # Build kwargs, excluding the _schedule key
    kwargs = {k: v for k, v in config.items() if k != "_schedule"}
    return ctor(**kwargs)


def reconstruct_opt_params(opt_params_serialized: dict[str, Any]) -> dict[str, Any]:
    """Walk a serialized ``opt_params`` dict and reconstruct any schedule objects.

    Values that are dicts with a ``"_schedule"`` key are reconstructed via
    ``reconstruct_schedule()``; everything else is left as-is.
    """
    result = {}
    for key, value in opt_params_serialized.items():
        if isinstance(value, dict) and "_schedule" in value:
            result[key] = reconstruct_schedule(value)
        else:
            result[key] = value
    return result


def _repo_exists(api: HfApi, repo_id: str, repo_type: str = "dataset") -> bool:
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        return True
    except Exception:
        return False


# ── Public API ──────────────────────────────────────────────────────────

def save_checkpoint_to_hf(
    save_dir: str,
    repo_id: str,
    phase: int,
    metadata: dict[str, Any] | None = None,
    first_layer: int = 23,
    total_steps: int = 30000,
    experimental_config: dict[str, Any] | None = None,
    opt_params: dict[str, Any] | None = None,
    token: str | None = None,
    repo_type: str = "dataset",
) -> str:
    """Zip a checkpoint directory and upload it to HuggingFace Hub.

    Args:
        save_dir:  Local directory with checkpoint files (e.g. ``./saved_titans_delta``).
        repo_id:   HF dataset repo, e.g. ``"veriga/titans-checkpoints"``.
        phase:     Training phase (1 or 2).  Used as filename prefix.
        metadata:  Dict of hyperparameters to store alongside the checkpoint.
        first_layer: Titans first active layer index (part of filename).
        total_steps:  Total training steps (part of filename).
        experimental_config: Training/experiment config dict (model arch, learning
            rate schedule, etc.) — serialised into metadata JSON.
        opt_params: Optimiser parameters dict (lr, weight decay, schedule, etc.)
            — serialised into metadata JSON.
        token:     HF API token.  Falls back to ``$HF_TOKEN`` env var.
        repo_type: ``"dataset"`` (default) or ``"model"``.

    Returns:
        The HF filename that was uploaded (e.g. ``"phase1_from_23_30000.zip"``).
    """
    hf_token = token or os.environ.get("HF_TOKEN")
    api = HfApi(token=hf_token)

    # Create repo if it doesn't exist
    if not _repo_exists(api, repo_id, repo_type):
        api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True, private=True)
        print(f"📦 Created HF repo: {repo_id}")

    # Build filenames
    stem = f"phase{phase}_from_{first_layer}_{total_steps}"
    zip_filename = f"{stem}.zip"
    meta_filename = f"{stem}_metadata.json"

    # Build full metadata — always include core training state
    full_meta: dict[str, Any] = dict(metadata) if metadata else {}
    full_meta["phase"] = phase
    full_meta["first_layer"] = first_layer
    full_meta["total_steps"] = total_steps
    if experimental_config is not None:
        full_meta["experimental_config"] = experimental_config
    if opt_params is not None:
        full_meta["opt_params"] = opt_params

    # Zip the checkpoint directory into a temp location
    tmp_dir = tempfile.mkdtemp()
    try:
        zip_path = shutil.make_archive(os.path.join(tmp_dir, stem), "zip", save_dir)

        # Write metadata JSON
        meta_path = os.path.join(tmp_dir, meta_filename)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(full_meta, f, indent=2, cls=_ScheduleEncoder, ensure_ascii=False)

        # Upload both files
        api.upload_file(
            path_or_fileobj=zip_path,
            path_in_repo=zip_filename,
            repo_id=repo_id,
            repo_type=repo_type,
        )
        api.upload_file(
            path_or_fileobj=meta_path,
            path_in_repo=meta_filename,
            repo_id=repo_id,
            repo_type=repo_type,
        )
        print(f"✅ Uploaded {zip_filename} + metadata → {repo_id}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return zip_filename


def load_checkpoint_from_hf(
    repo_id: str,
    phase: int,
    first_layer: int,
    total_steps: int,
    local_dir: str = ".",
    token: str | None = None,
    repo_type: str = "dataset",
) -> str | None:
    """Download & unzip a checkpoint from HuggingFace Hub.

    Args:
        repo_id:     HF dataset repo, e.g. ``"veriga/titans-checkpoints"``.
        phase:       Training phase (1 or 2).
        first_layer: Titans first active layer index.
        total_steps: Total training steps.
        local_dir:   Where to extract the checkpoint (default: cwd).
        token:       HF API token.  Falls back to ``$HF_TOKEN`` env var.
        repo_type:   ``"dataset"`` (default) or ``"model"``.

    Returns:
        Path to the extracted directory, or ``None`` if the checkpoint
        was not found on HF.
    """
    hf_token = token or os.environ.get("HF_TOKEN")

    stem = f"phase{phase}_from_{first_layer}_{total_steps}"
    zip_filename = f"{stem}.zip"
    extract_dir = os.path.join(local_dir, stem)

    # Skip if already extracted
    if os.path.isdir(extract_dir) and os.listdir(extract_dir):
        print(f"⏩ Checkpoint already extracted at {extract_dir}")
        return extract_dir

    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=zip_filename,
            repo_type=repo_type,
            token=hf_token,
        )
    except Exception as e:
        print(f"⚠️ Checkpoint {zip_filename} not found on HF: {e}")
        return None

    # Unzip
    shutil.unpack_archive(downloaded, extract_dir)
    print(f"✅ Extracted {zip_filename} → {extract_dir}")

    # Also download metadata if available
    try:
        meta_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{stem}_metadata.json",
            repo_type=repo_type,
            token=hf_token,
            local_dir=local_dir,
        )
        print(f"📋 Metadata saved to {meta_path}")
    except Exception:
        pass  # metadata is optional

    return extract_dir


def load_metadata_from_hf(
    repo_id: str,
    phase: int,
    first_layer: int,
    total_steps: int,
    token: str | None = None,
    repo_type: str = "dataset",
) -> dict[str, Any] | None:
    """Download only the metadata JSON for a checkpoint from HuggingFace Hub.

    Returns:
        Parsed metadata dict, or ``None`` if not found.
    """
    hf_token = token or os.environ.get("HF_TOKEN")
    stem = f"phase{phase}_from_{first_layer}_{total_steps}"
    meta_filename = f"{stem}_metadata.json"

    try:
        meta_path = hf_hub_download(
            repo_id=repo_id,
            filename=meta_filename,
            repo_type=repo_type,
            token=hf_token,
        )
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Metadata {meta_filename} not found on HF: {e}")
        return None


def save_last_metadata(
    repo_id: str,
    phase: int,
    first_layer: int,
    total_steps: int,
    experimental_config: dict[str, Any] | None = None,
    opt_params: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    token: str | None = None,
    repo_type: str = "dataset",
) -> str:
    """Upload a ``phase{N}_last_metadata.json`` pointing to the latest checkpoint.

    This file stores the hyperparameters of the **most recent** experiment for
    a given phase so that notebooks can auto-resume without hard-coding
    ``first_layer`` / ``total_steps``.

    Returns:
        The HF filename uploaded (e.g. ``"phase1_last_metadata.json"``).
    """
    hf_token = token or os.environ.get("HF_TOKEN")
    api = HfApi(token=hf_token)

    if not _repo_exists(api, repo_id, repo_type):
        api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True, private=True)

    filename = f"phase{phase}_last_metadata.json"
    payload: dict[str, Any] = dict(extra) if extra else {}
    payload["phase"] = phase
    payload["first_layer"] = first_layer
    payload["total_steps"] = total_steps
    payload["checkpoint"] = f"phase{phase}_from_{first_layer}_{total_steps}"
    if experimental_config is not None:
        payload["experimental_config"] = experimental_config
    if opt_params is not None:
        payload["opt_params"] = opt_params

    tmp_dir = tempfile.mkdtemp()
    try:
        local = os.path.join(tmp_dir, filename)
        with open(local, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, cls=_ScheduleEncoder, ensure_ascii=False)
        api.upload_file(
            path_or_fileobj=local,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type=repo_type,
        )
        print(f"✅ Uploaded {filename} → {repo_id}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return filename


def load_last_metadata(
    repo_id: str,
    phase: int,
    token: str | None = None,
    repo_type: str = "dataset",
) -> dict[str, Any] | None:
    """Download the ``phase{N}_last_metadata.json`` from HF.

    Returns:
        Parsed metadata dict with at least ``first_layer``, ``total_steps``,
        ``experimental_config``, ``opt_params``, or ``None`` if not found.
    """
    hf_token = token or os.environ.get("HF_TOKEN")
    filename = f"phase{phase}_last_metadata.json"

    try:
        local = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            token=hf_token,
        )
        with open(local, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ {filename} not found on HF: {e}")
        return None


def list_checkpoints(
    repo_id: str,
    phase: int | None = None,
    token: str | None = None,
    repo_type: str = "dataset",
) -> list[str]:
    """List checkpoint zip files in the HF repo, optionally filtered by phase."""
    hf_token = token or os.environ.get("HF_TOKEN")
    api = HfApi(token=hf_token)
    files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
    zips = [f for f in files if f.endswith(".zip")]
    if phase is not None:
        prefix = f"phase{phase}_"
        zips = [f for f in zips if f.startswith(prefix)]
    return sorted(zips)

"""
HuggingFace Hub checkpoint utilities for Titans training.

Phase 1 (per-layer distillation):
    Each Titans layer is trained independently with its own config and opt_params.
    Weights are saved as ``phase1_layer_{N}.zip``, metadata in a unified
    ``phase1_metadata.json`` keyed by ``layer_{N}``.

Phase 2 (joint LM fine-tuning):
    Uses ``phase2_last_metadata.json`` and ``phase{phase}_from_{first_layer}_{total_steps}.zip``.

Usage in notebooks:
    from hf_checkpoint import (
        save_phase1_layer_weights, load_phase1_layer_weights,
        load_phase1_metadata, save_phase1_metadata,
        save_phase1_last_metadata, load_phase1_last_metadata,
        load_all_phase1_layers,
        save_last_metadata, load_last_metadata,
        reconstruct_opt_params, schedule,
    )
"""

import json
import os
import shutil
import tempfile
from typing import Any

from huggingface_hub import HfApi, hf_hub_download, login


# ── Helpers ──────────────────────────────────────────────────────────────

# ── Schedule serialization ───────────────────────────────────────────────
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
    to ``save_phase1_layer_weights(opt_params=...)``.

    To get back a callable schedule, use ``reconstruct_schedule()`` or
    ``reconstruct_opt_params()``.
    """
    return {"_schedule": name, **kwargs}


def _callable_to_schedule_dict(fn) -> dict[str, Any]:
    """Introspect an optax schedule closure and convert to a serializable dict.

    optax schedules are closures that capture their constructor parameters as
    free variables.  This function extracts those variables and maps the
    signature to a known schedule type name so that ``reconstruct_schedule()``
    can rebuild the callable later.

    Detection strategy:
      1. Check ``__qualname__`` to identify composite schedules
         (``join_schedules`` = ``warmup_cosine_decay_schedule``).
      2. Check free-variable names to identify leaf schedules
         (``polynomial_schedule``, ``cosine_decay_schedule``, ``constant``).

    Supported schedules: ``warmup_cosine_decay``, ``polynomial``, ``linear``,
    ``cosine_decay``, ``constant``.
    """
    if not callable(fn):
        raise ValueError(f"Not callable: {fn!r}")
    if not hasattr(fn, "__closure__") or fn.__closure__ is None:
        raise ValueError(f"Not a closure: {fn!r}")

    qualname = getattr(fn, "__qualname__", "")

    # ── join_schedules (warmup_cosine_decay_schedule wrapper) ─────────
    # optax.warmup_cosine_decay_schedule creates a join_schedules with:
    #   boundaries = [warmup_steps]
    #   schedules  = [polynomial_schedule (warmup), cosine_decay_schedule]
    if "join_schedules" in qualname:
        freevars = fn.__code__.co_freevars
        values = [c.cell_contents for c in fn.__closure__]
        params = dict(zip(freevars, values))
        boundaries = params["boundaries"]
        inner_schedules = params["schedules"]
        if len(inner_schedules) == 2 and len(boundaries) == 1:
            warmup_fn = inner_schedules[0]
            cosine_fn = inner_schedules[1]
            w_freevars = warmup_fn.__code__.co_freevars
            w_values = [c.cell_contents for c in warmup_fn.__closure__]
            w_params = dict(zip(w_freevars, w_values))
            c_freevars = cosine_fn.__code__.co_freevars
            c_values = [c.cell_contents for c in cosine_fn.__closure__]
            c_params = dict(zip(c_freevars, c_values))
            # Reverse-engineer optax.warmup_cosine_decay_schedule params:
            # optax internally creates cosine_decay_schedule(decay_steps - warmup_steps),
            # so we must add warmup_steps back to get the original decay_steps.
            init_value = w_params["init_value"]
            peak_value = w_params["end_value"]
            warmup_steps = w_params["transition_steps"]
            decay_steps = c_params["decay_steps"] + warmup_steps
            alpha = c_params["alpha"]
            end_value = peak_value * alpha
            return {
                "_schedule": "warmup_cosine_decay",
                "init_value": init_value,
                "peak_value": peak_value,
                "warmup_steps": warmup_steps,
                "decay_steps": decay_steps,
                "end_value": end_value,
            }
        raise ValueError(f"Unsupported join_schedules structure: {params}")

    freevars = fn.__code__.co_freevars
    values = [c.cell_contents for c in fn.__closure__]
    params = dict(zip(freevars, values))
    param_keys = set(params.keys())

    # ── cosine_decay_schedule (alpha/exponent variant) ───────────────
    if "cosine_decay_schedule" in qualname:
        return {"_schedule": "cosine_decay", **params}

    # ── polynomial_schedule (used internally by linear_schedule) ─────
    if "polynomial_schedule" in qualname:
        return {"_schedule": "polynomial", **params}

    # ── constant_schedule ────────────────────────────────────────────
    if param_keys == {"value"}:
        return {"_schedule": "constant", **params}

    raise ValueError(
        f"Unknown schedule: qualname={qualname!r}, freevars={param_keys}"
    )


class _ScheduleEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy/JAX scalars and optax schedule callables.

    Callable values that look like optax schedules (closures with known
    parameter signatures) are automatically converted to serializable
    ``schedule()`` dicts with a ``"_schedule"`` key.
    """

    def default(self, obj: Any) -> Any:
        # numpy / JAX scalars
        if hasattr(obj, "item"):
            return obj.item()
        # optax schedule callables
        if callable(obj):
            try:
                return _callable_to_schedule_dict(obj)
            except (ValueError, AttributeError):
                pass
        # Fallback
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


def _ensure_repo(api: HfApi, repo_id: str, repo_type: str = "dataset") -> None:
    """Create the HF repo if it does not already exist."""
    if not _repo_exists(api, repo_id, repo_type):
        api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True, private=True)
        print(f"📦 Created HF repo: {repo_id}")


# ── Phase 1: Per-layer API ─────────────────────────────────────────────

def load_phase1_metadata(
    repo_id: str,
    token: str | None = None,
    repo_type: str = "dataset",
) -> dict[str, Any]:
    """Download the unified ``phase1_metadata.json`` from HF.

    Returns:
        Parsed metadata dict keyed by ``"layer_{N}"``, or an empty dict
        if the file does not exist yet.
    """
    hf_token = token or os.environ.get("HF_TOKEN")

    try:
        local = hf_hub_download(
            repo_id=repo_id,
            filename="phase1_metadata.json",
            repo_type=repo_type,
            token=hf_token,
        )
        with open(local, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ phase1_metadata.json not found on HF: {e}")
        return {}


def save_phase1_metadata(
    repo_id: str,
    metadata: dict[str, Any],
    token: str | None = None,
    repo_type: str = "dataset",
) -> str:
    """Upload (overwrite) the unified ``phase1_metadata.json`` to HF.

    Args:
        metadata: Full metadata dict keyed by ``"layer_{N}"``.

    Returns:
        The HF filename uploaded.
    """
    hf_token = token or os.environ.get("HF_TOKEN")
    api = HfApi(token=hf_token)
    _ensure_repo(api, repo_id, repo_type)

    filename = "phase1_metadata.json"
    tmp_dir = tempfile.mkdtemp()
    try:
        local = os.path.join(tmp_dir, filename)
        with open(local, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, cls=_ScheduleEncoder, ensure_ascii=False)
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


def save_phase1_layer_weights(
    layer_num: int,
    save_dir: str,
    repo_id: str,
    experimental_config: dict[str, Any] | None = None,
    opt_params: dict[str, Any] | None = None,
    warm_up: int = 0,
    total_steps: int = 30000,
    token: str | None = None,
    repo_type: str = "dataset",
) -> str:
    """Zip and upload weights for a **single** Titans layer, and update
    the unified ``phase1_metadata.json``.

    Args:
        layer_num: Titans layer index (e.g. 11, 17, 23).
        save_dir:  Local directory containing checkpoint files for this layer.
        repo_id:   HF dataset repo.
        experimental_config: Per-layer training config.
        opt_params: Per-layer optimiser parameters (serialisable).
        warm_up:   Warm-up steps for this layer.
        total_steps: Total training steps for this layer.
        token:     HF API token.
        repo_type: ``"dataset"`` (default) or ``"model"``.

    Returns:
        The HF zip filename uploaded (e.g. ``"phase1_layer_23.zip"``).
    """
    hf_token = token or os.environ.get("HF_TOKEN")
    api = HfApi(token=hf_token)
    _ensure_repo(api, repo_id, repo_type)

    zip_filename = f"phase1_layer_{layer_num}.zip"

    # ── Upload weights zip ────────────────────────────────────────────
    tmp_dir = tempfile.mkdtemp()
    try:
        zip_path = shutil.make_archive(
            os.path.join(tmp_dir, f"phase1_layer_{layer_num}"), "zip", save_dir,
        )
        api.upload_file(
            path_or_fileobj=zip_path,
            path_in_repo=zip_filename,
            repo_id=repo_id,
            repo_type=repo_type,
        )
        print(f"✅ Uploaded {zip_filename} → {repo_id}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── Update unified metadata ───────────────────────────────────────
    layer_key = f"layer_{layer_num}"
    metadata = load_phase1_metadata(repo_id, token=hf_token, repo_type=repo_type)
    metadata.setdefault(layer_key, {})
    if experimental_config is not None:
        metadata[layer_key]["experimental_config"] = experimental_config
    if opt_params is not None:
        metadata[layer_key]["opt_params"] = opt_params
    metadata[layer_key]["warm_up"] = warm_up
    metadata[layer_key]["total_steps"] = total_steps

    save_phase1_metadata(repo_id, metadata, token=hf_token, repo_type=repo_type)

    return zip_filename


def load_phase1_layer_weights(
    layer_num: int,
    repo_id: str,
    local_dir: str = ".",
    token: str | None = None,
    repo_type: str = "dataset",
) -> str | None:
    """Download & unzip weights for a **single** Titans layer from HF.

    Args:
        layer_num: Titans layer index.
        repo_id:   HF dataset repo.
        local_dir: Where to extract the checkpoint.
        token:     HF API token.
        repo_type: ``"dataset"`` or ``"model"``.

    Returns:
        Path to the extracted directory, or ``None`` if not found.
    """
    hf_token = token or os.environ.get("HF_TOKEN")

    zip_filename = f"phase1_layer_{layer_num}.zip"
    extract_dir = os.path.join(local_dir, f"phase1_layer_{layer_num}")

    if os.path.isdir(extract_dir) and os.listdir(extract_dir):
        print(f"⏩ Layer {layer_num} weights already extracted at {extract_dir}")
        return extract_dir

    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=zip_filename,
            repo_type=repo_type,
            token=hf_token,
        )
    except Exception as e:
        print(f"⚠️ {zip_filename} not found on HF: {e}")
        return None

    shutil.unpack_archive(downloaded, extract_dir)
    print(f"✅ Extracted {zip_filename} → {extract_dir}")
    return extract_dir


def load_all_phase1_layers(
    repo_id: str,
    titans_first_layer: int,
    local_dir: str = ".",
    token: str | None = None,
    repo_type: str = "dataset",
) -> dict[str, Any] | None:
    """Download weights for **all** trained Titans layers ≥ *titans_first_layer*.

    Reads ``phase1_metadata.json`` to discover which layers have been trained,
    downloads each ``phase1_layer_{N}.zip``, loads them via orbax, and returns
    a merged dict ``{"layer_N": params, ...}`` suitable for
    ``titans_tree_utils.merge_titans_params``.

    Args:
        repo_id:            HF dataset repo.
        titans_first_layer: Minimum layer index to include.
        local_dir:          Where to extract zips.
        token:              HF API token.
        repo_type:          ``"dataset"`` or ``"model"``.

    Returns:
        Combined titans params dict, or ``None`` if no layers found.
    """
    import orbax.checkpoint as ocp

    metadata = load_phase1_metadata(repo_id, token=token, repo_type=repo_type)
    if not metadata:
        print("⚠️ No phase1 metadata found — cannot load layers.")
        return None

    # Determine available layers >= titans_first_layer
    available_layers: list[int] = []
    for key in metadata:
        if key.startswith("layer_"):
            try:
                layer_idx = int(key.split("_", 1)[1])
                if layer_idx >= titans_first_layer:
                    available_layers.append(layer_idx)
            except ValueError:
                continue

    if not available_layers:
        print(f"⚠️ No trained layers ≥ {titans_first_layer} found in metadata.")
        return None

    available_layers.sort()
    print(f"📋 Found trained layers: {available_layers}")

    checkpointer = ocp.StandardCheckpointer()
    combined: dict[str, Any] = {}

    for layer_num in available_layers:
        layer_key = f"layer_{layer_num}"
        layer_dir = load_phase1_layer_weights(
            layer_num, repo_id, local_dir=local_dir,
            token=token, repo_type=repo_type,
        )
        if layer_dir is None:
            print(f"⚠️ Skipping {layer_key} — weights not found.")
            continue
        combined[layer_key] = checkpointer.restore(os.path.abspath(layer_dir))
        print(f"  ✅ Loaded {layer_key} weights")

    if not combined:
        return None

    return combined


# ── Phase 1: last-metadata (for resuming) ───────────────────────────────

def save_phase1_last_metadata(
    repo_id: str,
    target_layer: int,
    total_steps: int,
    warm_up: int = 0,
    experimental_config: dict[str, Any] | None = None,
    opt_params: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    token: str | None = None,
    repo_type: str = "dataset",
) -> str:
    """Upload ``phase1_last_metadata.json`` with the latest run info.

    This allows notebooks to resume a Phase 1 experiment without hard-coding
    parameters.

    Returns:
        The HF filename uploaded.
    """
    hf_token = token or os.environ.get("HF_TOKEN")
    api = HfApi(token=hf_token)
    _ensure_repo(api, repo_id, repo_type)

    filename = "phase1_last_metadata.json"
    payload: dict[str, Any] = dict(extra) if extra else {}
    payload["target_layer"] = target_layer
    payload["total_steps"] = total_steps
    payload["warm_up"] = warm_up
    payload["checkpoint"] = f"phase1_layer_{target_layer}"
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


def load_phase1_last_metadata(
    repo_id: str,
    token: str | None = None,
    repo_type: str = "dataset",
) -> dict[str, Any] | None:
    """Download ``phase1_last_metadata.json`` from HF.

    Returns:
        Parsed metadata dict, or ``None`` if not found.
    """
    hf_token = token or os.environ.get("HF_TOKEN")
    filename = "phase1_last_metadata.json"

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


# ── Phase 2: last-metadata ──────────────────────────────────────────────

def save_last_metadata(
    repo_id: str,
    phase: int,
    first_layer: int,
    total_steps: int,
    warm_up: int = 0,
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
        The HF filename uploaded (e.g. ``"phase2_last_metadata.json"``).
    """
    hf_token = token or os.environ.get("HF_TOKEN")
    api = HfApi(token=hf_token)
    _ensure_repo(api, repo_id, repo_type)

    filename = f"phase{phase}_last_metadata.json"
    payload: dict[str, Any] = dict(extra) if extra else {}
    payload["phase"] = phase
    payload["first_layer"] = first_layer
    payload["total_steps"] = total_steps
    payload["warm_up"] = warm_up
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


# ── Phase 2: Combined checkpoint save/load ────────────────────────────────

def save_checkpoint_to_hf(
    save_dir: str,
    repo_id: str,
    phase: int,
    experimental_config: dict[str, Any] | None = None,
    opt_params: dict[str, Any] | None = None,
    first_layer: int = 0,
    total_steps: int = 0,
    warm_up: int = 0,
    token: str | None = None,
    repo_type: str = "dataset",
) -> str:
    """Zip a local checkpoint directory and upload it to HF.

    The zip filename follows the pattern
    ``phase{phase}_from_{first_layer}_{total_steps}.zip``.

    Returns:
        The HF zip filename uploaded.
    """
    hf_token = token or os.environ.get("HF_TOKEN")
    api = HfApi(token=hf_token)
    _ensure_repo(api, repo_id, repo_type)

    zip_stem = f"phase{phase}_from_{first_layer}_{total_steps}"
    zip_filename = f"{zip_stem}.zip"

    tmp_dir = tempfile.mkdtemp()
    try:
        zip_path = shutil.make_archive(
            os.path.join(tmp_dir, zip_stem), "zip", save_dir,
        )
        api.upload_file(
            path_or_fileobj=zip_path,
            path_in_repo=zip_filename,
            repo_id=repo_id,
            repo_type=repo_type,
        )
        print(f"✅ Uploaded {zip_filename} → {repo_id}")
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
    """Download & unzip a phase checkpoint from HF.

    Returns:
        Path to the extracted directory, or ``None`` if not found.
    """
    hf_token = token or os.environ.get("HF_TOKEN")

    zip_filename = f"phase{phase}_from_{first_layer}_{total_steps}.zip"
    extract_dir = os.path.join(local_dir, f"phase{phase}_from_{first_layer}_{total_steps}")

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
        print(f"⚠️ {zip_filename} not found on HF: {e}")
        return None

    shutil.unpack_archive(downloaded, extract_dir)
    print(f"✅ Extracted {zip_filename} → {extract_dir}")
    return extract_dir


# ── Training Reports ────────────────────────────────────────────────────

def save_training_report(
    repo_id: str,
    phase: int,
    first_layer: int,
    total_steps: int,
    loss_history: list[dict[str, Any]],
    extra_metrics: dict[str, Any] | None = None,
    experimental_config: dict[str, Any] | None = None,
    opt_params: dict[str, Any] | None = None,
    token: str | None = None,
    repo_type: str = "dataset",
) -> str:
    """Upload a training report (JSON + loss plot PNG) to HF.

    The report ties training results to the hyperparameters that produced them.

    Args:
        repo_id: HF dataset repo.
        phase: Training phase (1 or 2).
        first_layer: First Titans layer index.
        total_steps: Total training steps completed.
        loss_history: List of ``{"step": int, "value": float}`` dicts.
        extra_metrics: Dict of extra metrics (e.g. last 500 avg loss, accuracy).
        experimental_config: Model config used.
        opt_params: Optimizer params used.
        token: HF API token.
        repo_type: ``"dataset"`` or ``"model"``.

    Returns:
        The report filename uploaded (e.g. ``"phase2_report_from_11_60000.json"``).
    """
    hf_token = token or os.environ.get("HF_TOKEN")
    api = HfApi(token=hf_token)
    _ensure_repo(api, repo_id, repo_type)

    stem = f"phase{phase}_report_from_{first_layer}_{total_steps}"

    # ── Compute loss statistics ──────────────────────────────────────
    import numpy as _np

    if loss_history:
        values = [e["value"] for e in loss_history]
        steps = [e["step"] for e in loss_history]
        stats = {
            "loss_min": float(min(values)),
            "loss_max": float(max(values)),
            "loss_final": float(values[-1]),
            "loss_mean": float(_np.mean(values)),
            "loss_std": float(_np.std(values)),
        }
        # Last N steps average
        for n in (100, 500, 1000):
            if len(values) >= n:
                stats[f"loss_last_{n}_mean"] = float(_np.mean(values[-n:]))
    else:
        stats = {}

    # ── Build JSON report ────────────────────────────────────────────
    report: dict[str, Any] = {
        "phase": phase,
        "first_layer": first_layer,
        "total_steps": total_steps,
        "checkpoint": f"phase{phase}_from_{first_layer}_{total_steps}",
        "loss_stats": stats,
        "loss_history_sample": loss_history,  # full history or sampled
    }
    if extra_metrics:
        report["extra_metrics"] = extra_metrics
    if experimental_config:
        report["experimental_config"] = experimental_config
    if opt_params:
        report["opt_params"] = opt_params

    tmp_dir = tempfile.mkdtemp()
    try:
        # ── Upload JSON report ───────────────────────────────────────
        json_filename = f"{stem}.json"
        json_path = os.path.join(tmp_dir, json_filename)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, cls=_ScheduleEncoder, ensure_ascii=False)
        api.upload_file(
            path_or_fileobj=json_path,
            path_in_repo=json_filename,
            repo_id=repo_id,
            repo_type=repo_type,
        )
        print(f"✅ Uploaded {json_filename} → {repo_id}")

        # ── Upload loss plot PNG ─────────────────────────────────────
        if loss_history:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(steps, values, linewidth=0.5, alpha=0.7, label="loss")
                if "loss_last_500_mean" in stats:
                    last_500_avg = stats["loss_last_500_mean"]
                    ax.axhline(
                        y=last_500_avg, color="r", linestyle="--",
                        label=f"last 500 avg: {last_500_avg:.4f}",
                    )
                ax.set_xlabel("Step")
                ax.set_ylabel("Loss")
                ax.set_title(
                    f"Phase {phase} Loss (layers ≥{first_layer}, {total_steps} steps)"
                )
                ax.legend()
                ax.grid(True, alpha=0.3)
                fig.tight_layout()

                png_filename = f"{stem}.png"
                png_path = os.path.join(tmp_dir, png_filename)
                fig.savefig(png_path, dpi=150)
                plt.close(fig)

                api.upload_file(
                    path_or_fileobj=png_path,
                    path_in_repo=png_filename,
                    repo_id=repo_id,
                    repo_type=repo_type,
                )
                print(f"✅ Uploaded {png_filename} → {repo_id}")
            except ImportError:
                print("⚠️ matplotlib not available — skipping loss plot upload")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return stem


def load_training_report(
    repo_id: str,
    phase: int,
    first_layer: int,
    total_steps: int,
    token: str | None = None,
    repo_type: str = "dataset",
) -> dict[str, Any] | None:
    """Download a training report JSON from HF.

    Returns:
        Parsed report dict, or ``None`` if not found.
    """
    hf_token = token or os.environ.get("HF_TOKEN")
    filename = f"phase{phase}_report_from_{first_layer}_{total_steps}.json"

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


def read_tensorboard_losses(
    workdir: str,
    tag: str = "lm_loss",
) -> list[dict[str, Any]]:
    """Read loss values from TensorBoard event files in a Kauldron workdir.

    Args:
        workdir: Path to the Kauldron workdir (e.g. ``titans_workdir_phase2_from11``).
        tag: The TensorBoard tag to read (default ``"lm_loss"``).

    Returns:
        List of ``{"step": int, "value": float}`` sorted by step.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        print("⚠️ tensorboard not installed — cannot read event files")
        return []

    summaries_dir = os.path.join(workdir, "summaries")
    if not os.path.isdir(summaries_dir):
        # Try top-level workdir
        summaries_dir = workdir

    ea = EventAccumulator(summaries_dir)
    ea.Reload()

    # Find the tag (Kauldron may prefix it)
    available = ea.Tags().get("scalars", [])
    matched_tag = None
    for t in available:
        if tag in t:
            matched_tag = t
            break
    if matched_tag is None and available:
        print(f"⚠️ Tag '{tag}' not found. Available: {available[:10]}")
        return []

    events = ea.Scalars(matched_tag)
    return [{"step": e.step, "value": e.value} for e in events]


# ── Utility ─────────────────────────────────────────────────────────────

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
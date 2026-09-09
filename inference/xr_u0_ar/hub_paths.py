from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable


TOKENIZER_ALLOW_PATTERNS = (
    "*.json",
    "*.py",
    "*.tiktoken",
    "*.txt",
    "*_tokens.txt",
    "README.md",
)


def _snapshot_download(**kwargs: Any) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(**kwargs)


def _env_bool(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None:
        return None
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _clean(value: str | os.PathLike[str] | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _looks_like_missing_local_path(value: str) -> bool:
    if value.startswith(("/", "./", "../", "~")):
        return True
    return value.count("/") > 1


def hub_kwargs_from_config(config: Any) -> dict[str, Any]:
    """Extract optional HuggingFace path-resolution settings from a config."""
    getter = config.get if isinstance(config, dict) else lambda key, default=None: getattr(config, key, default)
    return {
        "revision": getter("hf_revision", None),
        "cache_dir": getter("hf_cache_dir", None),
        "local_files_only": getter("hf_local_files_only", None),
    }


def resolve_hub_or_local(
    path: str | os.PathLike[str],
    *,
    kind: str,
    allow_patterns: Iterable[str] | None = None,
    revision: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    local_files_only: bool | None = None,
) -> str:
    """Return a local directory for either an existing path or a Hub repo ID."""
    raw = _clean(path)
    if raw is None:
        raise ValueError(f"{kind} path is empty")

    local = Path(raw).expanduser()
    if local.exists():
        return str(local.resolve())

    if _looks_like_missing_local_path(raw):
        raise FileNotFoundError(f"{kind} local path does not exist: {raw}")

    resolved_revision = _clean(revision) or _clean(os.environ.get("XR_U0_HF_REVISION"))
    resolved_cache_dir = _clean(cache_dir) or _clean(os.environ.get("XR_U0_HF_CACHE_DIR"))
    resolved_local_files_only = (
        local_files_only
        if local_files_only is not None
        else (_env_bool("XR_U0_HF_LOCAL_FILES_ONLY") or False)
    )

    kwargs: dict[str, Any] = {
        "repo_id": raw,
        "local_files_only": bool(resolved_local_files_only),
    }
    if allow_patterns is not None:
        kwargs["allow_patterns"] = list(allow_patterns)
    if resolved_revision:
        kwargs["revision"] = resolved_revision
    if resolved_cache_dir:
        kwargs["cache_dir"] = resolved_cache_dir

    try:
        return _snapshot_download(**kwargs)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "automatic HuggingFace Hub downloads require the `huggingface-hub` "
            "package. Install project dependencies with `pip install -e .`."
        ) from exc
    except Exception as exc:
        mode = "local cache" if resolved_local_files_only else "HuggingFace Hub"
        raise RuntimeError(
            f"failed to resolve {kind} path '{raw}' from {mode}. "
            "Use an existing local path, run `huggingface-cli login` or set "
            "`HF_TOKEN` for gated repositories, and check the repo ID."
        ) from exc


def resolve_model_path(path: str | os.PathLike[str], **kwargs: Any) -> str:
    return resolve_hub_or_local(path, kind="model", **kwargs)


def resolve_tokenizer_path(path: str | os.PathLike[str], **kwargs: Any) -> str:
    return resolve_hub_or_local(
        path,
        kind="tokenizer",
        allow_patterns=TOKENIZER_ALLOW_PATTERNS,
        **kwargs,
    )


def resolve_vision_tokenizer_path(path: str | os.PathLike[str], **kwargs: Any) -> str:
    return resolve_hub_or_local(path, kind="VisionTokenizer", **kwargs)


def resolve_local_file(
    path: str | os.PathLike[str],
    *,
    kind: str,
    base_dir: str | os.PathLike[str] | None = None,
) -> str:
    """Resolve a local file path, optionally relative to a resolved base dir."""
    raw = _clean(path)
    if raw is None:
        raise ValueError(f"{kind} path is empty")

    candidates: list[Path] = []
    given = Path(raw).expanduser()
    if given.is_absolute():
        candidates.append(given)
    else:
        if base_dir is not None:
            candidates.append(Path(base_dir).expanduser() / given)
        candidates.append(given)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())

    tried = ", ".join(str(item) for item in candidates)
    raise FileNotFoundError(f"{kind} file does not exist: {raw} (tried: {tried})")

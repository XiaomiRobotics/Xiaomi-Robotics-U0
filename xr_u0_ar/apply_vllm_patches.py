#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REQUIRED_VLLM_VERSION = "0.11.0"
REPO_ROOT = Path(__file__).resolve().parents[1]
PATCH_DIR = Path(__file__).resolve().parent / "third_party" / "vllm"


def get_vllm_site() -> Path:
    try:
        import vllm
    except ImportError:
        sys.exit("[error] vllm is not installed. Install vllm==0.11.0 first.")
    if vllm.__version__ != REQUIRED_VLLM_VERSION:
        sys.exit(
            f"[fatal] vLLM version must be {REQUIRED_VLLM_VERSION}; "
            f"found {vllm.__version__}."
        )
    print(f"[info] vLLM {vllm.__version__} detected")
    return Path(vllm.__file__).parent


def run_patch(patch_file: Path, site_dir: Path, *, dry_run: bool = False, reverse: bool = False) -> tuple[bool, str]:
    cmd = ["patch", "-p2", "--silent"]
    if dry_run:
        cmd.append("--dry-run")
    if reverse:
        cmd.append("-R")
    with patch_file.open() as f:
        result = subprocess.run(
            cmd,
            cwd=str(site_dir),
            stdin=f,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    return result.returncode == 0, result.stdout + result.stderr


def patch_state(patch_file: Path, site_dir: Path) -> str:
    rel = patch_file.relative_to(PATCH_DIR).as_posix()
    if rel == "model_executor/models/registry.py.patch":
        registry = site_dir / "model_executor" / "models" / "registry.py"
        if registry.exists() and '"UNISForCausalLM": ("unis", "UNISForCausalLM")' in registry.read_text(encoding="utf-8"):
            return "applied"
    cmd = ["patch", "-p2", "--forward", "--dry-run", "--silent"]
    with patch_file.open() as f:
        result = subprocess.run(
            cmd,
            cwd=str(site_dir),
            stdin=f,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    out = (result.stdout + result.stderr).lower()
    if result.returncode == 0:
        return "unapplied"
    if "previously applied" in out or "reversed (or previously applied)" in out:
        return "applied"
    if "already exists" in out and "creating file" not in out:
        return "applied"
    return "broken"


def collect_patch_targets(patch_file: Path) -> list[str]:
    targets: set[str] = set()
    with patch_file.open() as f:
        for line in f:
            if not line.startswith(("--- a/vllm/", "+++ b/vllm/")):
                continue
            rel = line.split("\t")[0].split(" ", 1)[-1].strip()
            if rel.endswith("/dev/null"):
                continue
            targets.add(rel[len("a/vllm/"):])
    return sorted(targets)


def _replace_once(path: Path, needle: str, replacement: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if replacement in text:
        return False
    if needle not in text:
        raise RuntimeError(f"Could not find expected insertion point in {path}")
    path.write_text(text.replace(needle, replacement, 1), encoding="utf-8")
    return True


def _insert_after_once(path: Path, anchor: str, insertion: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if insertion in text:
        return False
    if anchor not in text:
        raise RuntimeError(f"Could not find expected insertion point in {path}")
    path.write_text(text.replace(anchor, anchor + insertion, 1), encoding="utf-8")
    return True


def register_unis_config(site_dir: Path) -> None:
    config_py = site_dir / "transformers_utils" / "config.py"
    init_py = site_dir / "transformers_utils" / "configs" / "__init__.py"
    _insert_after_once(
        config_py,
        'step3_text="Step3TextConfig",\n',
        '    UNIS="UNISConfig",\n',
    )
    _insert_after_once(
        init_py,
        "from vllm.transformers_utils.configs.radio import RadioConfig\n",
        "from vllm.transformers_utils.configs.unis import UNISConfig\n",
    )
    _insert_after_once(
        init_py,
        '    "Step3TextConfig",\n',
        '    "UNISConfig",\n',
    )


def cmd_apply(site_dir: Path) -> None:
    patches = sorted(PATCH_DIR.rglob("*.patch"))
    if not patches:
        sys.exit(f"[error] no patch files under {PATCH_DIR.relative_to(REPO_ROOT)}")
    states = [(p, patch_state(p, site_dir)) for p in patches]
    broken = [p for p, state in states if state == "broken"]
    if broken:
        for p in broken:
            print(f"[broken] {p.relative_to(PATCH_DIR)}")
        sys.exit("[fatal] some patches do not apply cleanly. Reinstall vLLM and retry.")

    backup_root = site_dir.parent / "vllm_unis_ar_patch_backup"
    backup_root.mkdir(exist_ok=True)
    for patch_file, state in states:
        if state != "unapplied":
            continue
        for rel in collect_patch_targets(patch_file):
            src = site_dir / rel
            if src.exists():
                dst = backup_root / rel
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)

    applied = 0
    for patch_file, state in states:
        if state != "unapplied":
            continue
        ok, log = run_patch(patch_file, site_dir)
        if not ok:
            sys.exit(f"[fatal] failed to apply {patch_file.relative_to(PATCH_DIR)}\n{log}")
        applied += 1
        print(f"[applied] {patch_file.relative_to(PATCH_DIR)}")
    register_unis_config(site_dir)
    print(f"[ok] AR patch flow complete; applied {applied} patch(es)")


def cmd_revert(site_dir: Path) -> None:
    backup_root = site_dir.parent / "vllm_unis_ar_patch_backup"
    if not backup_root.exists():
        sys.exit(f"[error] no backup at {backup_root}")
    count = 0
    for src in backup_root.rglob("*"):
        if not src.is_file():
            continue
        dst = site_dir / src.relative_to(backup_root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        count += 1
    print(f"[ok] restored {count} file(s) from {backup_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply Xiaomi-Robotics-U0-AR vLLM 0.11.0 patches.")
    parser.add_argument("--revert", action="store_true")
    args = parser.parse_args()
    site_dir = get_vllm_site()
    if args.revert:
        cmd_revert(site_dir)
    else:
        cmd_apply(site_dir)


if __name__ == "__main__":
    main()

"""
Fresh-clone smoke test.

A fresh ``git clone`` of this repository does **not** contain the private system
workbook or any derived table (they are all gitignored), so the notebooks cannot
run out of the box. This script proves that they *can* run end to end from a
clean tree using only the synthetic workbook generator:

  1. copy the working tree to a throw-away temporary directory, excluding
     ``.git`` and **all** private data (raw workbook, derived tables, model
     bundles, result tables/figures, and every ``.csv``/``.xlsx`` copy);
  2. generate a synthetic ``data/raw/synthetic_channel_data.xlsx`` inside the copy;
  3. execute notebooks 01-08 in order with a per-notebook timeout;
  4. verify that the expected processed tables, model bundles and report
     artifacts are regenerated.

The real repository and the real workbook are never read from or written to.
Run from the repository root:

    python tests/fresh_clone_smoke.py                 # all notebooks
    python tests/fresh_clone_smoke.py --notebooks 1-3 # a subset
    python tests/fresh_clone_smoke.py --prepare-only  # build the tree, stop
    python tests/fresh_clone_smoke.py --keep          # keep the temp tree

Exit code 0 means success (or a graceful skip when the tooling is unavailable).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve()
SRC = HERE.parent.parent  # repository root (tests/ -> repo)

# directories never copied into the temp clone
IGNORE_DIR_NAMES = {".git", "__pycache__", ".ipynb_checkpoints", ".venv",
                    ".mypy_cache", ".pytest_cache", ".ipynb_checkpoints"}
PRIVATE_PREFIXES = ("data/raw", "data/interim", "data/processed", "models",
                    "reports/tables", "reports/figures")
PRIVATE_SUFFIXES = (".xlsx", ".xls", ".csv")

# artifacts that must exist after a successful end-to-end run
EXPECTED_ARTIFACTS = [
    "data/interim/channel_dictionary.csv",
    "data/interim/channel_values.csv",
    "data/interim/channel_clean.csv",
    "data/interim/channel_scaled.csv",
    "data/interim/state_segments.csv",
    "data/processed/features_labels.csv",
    "data/processed/feature_list.csv",
    "data/processed/label_config.json",
    "models/05_logreg_early_warning.joblib",
    "models/05_tree_early_warning.joblib",
    "reports/tables/08_data_availability.csv",
    "reports/tables/08_alarm_episodes_union.csv",
    "reports/figures/notebook_08_plot_01.png",
]


def _ignore(dirpath, names):
    out = set()
    dp = Path(dirpath)
    for name in names:
        p = dp / name
        if name in IGNORE_DIR_NAMES or p.suffix.lower() in PRIVATE_SUFFIXES:
            out.add(name)
            continue
        rel = p.relative_to(SRC).as_posix()
        if any(rel == pref or rel.startswith(pref + "/") for pref in PRIVATE_PREFIXES):
            out.add(name)
    return out


def _parse_selector(selector: str, notebooks: list[Path]) -> list[Path]:
    if not selector or selector.lower() in ("all", "*"):
        return notebooks
    wanted = set()
    for part in selector.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            wanted.update(range(int(a), int(b) + 1))
        elif part:
            wanted.add(int(part))
    picked = [nb for nb in notebooks if int(nb.name[:2]) in wanted]
    if not picked:
        raise SystemExit(f"--notebooks {selector!r} matched no notebook")
    return picked


def _run(cmd, cwd, env, timeout):
    started = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, timeout=timeout,
                              capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        return None, f"TIMEOUT after {timeout}s", time.time() - started
    ok = proc.returncode == 0
    out = (proc.stdout or "") + (proc.stderr or "")
    return ok, out, time.time() - started


def _notebook_error(nb_path: Path) -> str:
    """Pull the first error output out of an executed notebook, if any."""
    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return ""
    for cell in nb.get("cells", []):
        for o in cell.get("outputs", []):
            if o.get("output_type") == "error":
                return f"{o.get('ename')}: {o.get('evalue')}"
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--notebooks", default="all", help="e.g. all, 1-3, 1,3,8")
    ap.add_argument("--timeout", type=int, default=900, help="per-notebook seconds")
    ap.add_argument("--prepare-only", action="store_true",
                    help="only build the temp tree and synthetic workbook")
    ap.add_argument("--keep", action="store_true", help="keep the temp directory")
    ap.add_argument("--workdir", default=str(SRC), help="source tree to copy")
    args = ap.parse_args()

    if importlib.util.find_spec("nbconvert") is None:
        print("SKIP fresh_clone_smoke.py: nbconvert is not installed "
              "(pip install -r requirements.txt)")
        return 0

    src = Path(args.workdir).resolve()
    if not (src / "notebooks").is_dir():
        raise SystemExit(f"not a project tree: {src}")

    notebooks = sorted((src / "notebooks").glob("*.ipynb"))
    notebooks = [n for n in notebooks if n.name[:2].isdigit()]
    selected = _parse_selector(args.notebooks, notebooks)

    tmp = Path(tempfile.mkdtemp(prefix="pm_smoke_"))
    print(f"temp clone: {tmp}")
    try:
        shutil.copytree(src, tmp / "clone", ignore=_ignore, dirs_exist_ok=True)
        clone = tmp / "clone"
        for sub in ("data/raw", "data/interim", "data/processed", "models",
                    "reports/tables", "reports/figures"):
            (clone / sub).mkdir(parents=True, exist_ok=True)

        env = dict(os.environ)
        env["PYTHONPATH"] = str(clone) + os.pathsep + env.get("PYTHONPATH", "")
        env["MPLBACKEND"] = "Agg"
        env["PYTHONIOENCODING"] = "utf-8"

        print(f"[0/0] generating synthetic workbook ...")
        ok, out, secs = _run([sys.executable, "tests/generate_synthetic_channels.py",
                              "--root", str(clone)], clone, env, timeout=120)
        if not ok:
            print("FAILED to generate synthetic workbook:\n" + out[-4000:])
            return 1
        print(f"      synthetic workbook ready ({secs:.1f}s)")

        if args.prepare_only:
            print("prepare-only: stopping before notebook execution")
            return 0

        for idx, nb in enumerate(selected, start=1):
            rel = nb.relative_to(src).as_posix()
            print(f"[{idx}/{len(selected)}] executing {rel} "
                  f"(timeout {args.timeout}s) ...")
            cmd = [sys.executable, "-m", "nbconvert", "--to", "notebook",
                   "--execute", "--inplace",
                   f"--ExecutePreprocessor.timeout={args.timeout}", rel]
            ok, out, secs = _run(cmd, clone, env, timeout=args.timeout + 120)
            if not ok:
                detail = _notebook_error(clone / rel) or out[-4000:]
                print(f"FAILED {rel} after {secs:.0f}s\n{detail}")
                return 1
            print(f"      ok ({secs:.0f}s)")

        if len(selected) != len(notebooks):
            print(f"\nran {len(selected)}/{len(notebooks)} notebooks; "
                  "skipping full-artifact verification")
            print("subset smoke run PASSED")
            return 0

        print("\nverifying regenerated artifacts ...")
        missing = [a for a in EXPECTED_ARTIFACTS if not (clone / a).is_file()]
        if missing:
            print("FAILED: missing artifacts:\n  " + "\n  ".join(missing))
            return 1
        print(f"all {len(EXPECTED_ARTIFACTS)} expected artifacts regenerated")
        print("fresh-clone smoke test PASSED")
        return 0
    finally:
        if args.keep:
            print(f"kept temp tree: {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python
"""Scan local HF cache and report artifact inspection results.

This is a throwaway utility for local debugging. It walks the Hugging Face Hub
cache (`HF_HUB_CACHE`) and, for each repo snapshot, runs the artifact inspector
to extract family/component/confidence/evidence without loading tensors.
"""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import constants

from nodetool.integrations.huggingface.artifact_inspector import inspect_paths


def main() -> None:
    cache_root = Path(constants.HF_HUB_CACHE)
    if not cache_root.exists():
        print(f"HF cache not found at {cache_root}")
        return

    repos = sorted(p for p in cache_root.iterdir() if p.is_dir())
    results = []

    for repo_dir in repos:
        snapshots = repo_dir / "snapshots"
        if not snapshots.exists():
            continue
        for snapshot in sorted(snapshots.iterdir()):
            if not snapshot.is_dir():
                continue
            files = [str(p) for p in snapshot.rglob("*") if p.is_file()]
            detection = inspect_paths(files)
            results.append(
                {
                    "repo_dir": repo_dir.name,
                    "snapshot": snapshot.name,
                    "artifact_family": getattr(detection, "family", None),
                    "artifact_component": getattr(detection, "component", None),
                    "artifact_confidence": getattr(detection, "confidence", None),
                    "artifact_evidence": getattr(detection, "evidence", None),
                }
            )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

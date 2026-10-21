#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def run_curl(url: str, part: Path, start: int, end: int, expected: int):
    if part.exists() and part.stat().st_size == expected:
        return str(part), "skip"
    if part.exists():
        part.unlink()
    cmd = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        "30",
        "--retry-delay",
        "3",
        "--connect-timeout",
        "30",
        "--speed-time",
        "60",
        "--speed-limit",
        "1024",
        "-r",
        f"{start}-{end}",
        "-o",
        str(part),
        url,
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"curl failed for {part.name}: rc={proc.returncode}\n{proc.stdout[-4000:]}")
    got = part.stat().st_size
    if got != expected:
        raise RuntimeError(f"bad part size for {part.name}: got={got} expected={expected}")
    return str(part), "downloaded"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-mib", type=int, default=512)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size == args.size:
        print(f"[OK] already complete: {out}", flush=True)
        return

    part_dir = out.parent / f".{out.name}.parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    chunk = args.chunk_mib * 1024 * 1024
    ranges = []
    pos = 0
    idx = 0
    while pos < args.size:
        end = min(pos + chunk - 1, args.size - 1)
        expected = end - pos + 1
        ranges.append((idx, pos, end, expected, part_dir / f"part-{idx:04d}"))
        pos = end + 1
        idx += 1

    print(f"[INFO] ranges={len(ranges)} workers={args.workers} out={out}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_curl, args.url, part, start, end, expected) for idx, start, end, expected, part in ranges]
        for fut in as_completed(futs):
            part, status = fut.result()
            print(f"[PART] {status} {part}", flush=True)

    tmp = out.with_suffix(out.suffix + ".merged")
    with tmp.open("wb") as w:
        for idx, start, end, expected, part in ranges:
            if part.stat().st_size != expected:
                raise RuntimeError(f"part changed before merge: {part}")
            with part.open("rb") as r:
                shutil.copyfileobj(r, w, length=16 * 1024 * 1024)
    got = tmp.stat().st_size
    if got != args.size:
        raise RuntimeError(f"bad merged size: got={got} expected={args.size}")
    os.replace(tmp, out)
    print(f"[OK] merged {out} size={out.stat().st_size}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        raise

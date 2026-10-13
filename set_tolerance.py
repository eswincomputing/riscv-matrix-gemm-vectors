#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Beijing ESWIN Computing Technology Co., Ltd.
"""Set the tolerance used by the on-target self-check of the error-bound cases.

The programs ship with G = K, which every FP32-accumulating implementation meets.
If you know how often your implementation rounds, tighten the self-check:

    python3 set_tolerance.py all --group 16     # rounds once per 16 products: G = ceil(K/16)
    python3 set_tolerance.py 004 --rounds 8     # explicit G for one case
    python3 set_tolerance.py all --reset        # back to G = K

The tolerance is the binary64 value gamma(G) = G*u / (1 - G*u), u = 2^-24, stored in
the mailbox word "tol" (see meta.json). This script rewrites it in prog.hex and
prog.elf in place; no toolchain is needed. Bit-exact cases are left unchanged.
"""

import argparse
import json
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def gamma(G, u=2.0 ** -24):
    return G * u / (1.0 - G * u)


def patch(d, G):
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    if meta["criterion"]["type"] != "error-bound":
        return None
    base = int(json.loads((HERE / "index.json").read_text(encoding="utf-8"))["base"], 16)
    tol_addr = int(meta["mailbox"]["tol"], 16)
    new = struct.pack("<d", gamma(G))
    lo, hi = struct.unpack("<II", new)

    hexf = d / "prog.hex"
    lines = hexf.read_text(encoding="ascii").splitlines()
    head = 1 if lines and lines[0].startswith("//") else 0
    i = head + (tol_addr - base) // 4
    old = struct.unpack("<d", struct.pack("<II", int(lines[i], 16), int(lines[i + 1], 16)))[0]
    lines[i], lines[i + 1] = f"{lo:08x}", f"{hi:08x}"
    hexf.write_text("\n".join(lines) + "\n", encoding="ascii")

    elf = d / "prog.elf"
    if elf.exists():
        data = bytearray(elf.read_bytes())
        # The mailbox starts with done=0, status=0, fail_index=-1, fail_count=0, then tol.
        prefix = struct.pack("<IIiI", 0, 0, -1, 0)
        pos = data.find(prefix + struct.pack("<d", old))
        if pos < 0 or data.find(prefix + struct.pack("<d", old), pos + 1) >= 0:
            sys.exit(f"{d.name}: could not locate the tolerance in prog.elf uniquely")
        data[pos + 16:pos + 24] = new
        elf.write_bytes(bytes(data))
    return old, gamma(G)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("case", help="case id, or 'all'")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--rounds", type=int, help="rounding count G")
    g.add_argument("--group", type=int, help="the implementation rounds once per GROUP products")
    g.add_argument("--reset", action="store_true", help="restore G = K")
    args = ap.parse_args()
    dirs = sorted((HERE / "cases").iterdir())
    if args.case != "all":
        dirs = [d for d in dirs if d.name.startswith(args.case)]
        if not dirs:
            sys.exit(f"no case {args.case}")
    for d in dirs:
        K = json.loads((d / "meta.json").read_text(encoding="utf-8"))["shape"]["K"]
        G = K if args.reset else (args.rounds if args.rounds else -(-K // args.group))
        r = patch(d, G)
        if r is None:
            print(f"  {d.name:26s} bit-exact, unchanged")
        else:
            print(f"  {d.name:26s} G = {G:<5d} tol {r[0]:.6e} -> {r[1]:.6e}")


if __name__ == "__main__":
    main()

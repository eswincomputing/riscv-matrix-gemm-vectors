#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Beijing ESWIN Computing Technology Co., Ltd.
"""Self-test of the vectors and of check.py (needs numpy; check.py itself does not).

    python3 tests/selftest.py              # all checks
    python3 tests/selftest.py --skip-regen # skip the regeneration check

1. Reproducibility: the generator, run into a temporary directory, reproduces every
   committed data file, program source, linker script and meta.json byte for byte.
2. Compliant implementations pass: several accumulation schemes that all accumulate in
   FP32 with round-to-nearest-even (sequential, reversed, groups of 16 or 32 exact
   products, pairwise tree) pass every case with the default G = K and with their own
   tighter G; on the bit-exact cases they all produce exactly the expected bits.
3. Faults are caught: one element off by 1% (at the output with the least
   cancellation, where the bound is tightest), the last instruction missing, the
   accumulator not cleared, A and B swapped, and for the bit-exact cases truncation,
   ties-away rounding and a 23-bit accumulator.
"""

import argparse
import filecmp
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import check  # noqa: E402

F32 = np.float32


# ─────────────────────────── data ───────────────────────────

def bf16_matrix(path, rows, cols, stride):
    w = np.fromfile(path, dtype="<u2").reshape(rows, stride)[:, :cols]
    return (w.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def load_ab(case):
    m = case.meta
    sa, sb = m["stride_bytes"]["A"] // 2, m["stride_bytes"]["B"] // 2
    return (bf16_matrix(case.dir / "a.bin", case.M, case.K, sa),
            bf16_matrix(case.dir / "b.bin", case.N, case.K, sb))


# ─────────────────────────── rounding helpers ───────────────────────────

def r_rne(x):
    return x.astype(F32)


def r_rtz(x):
    y = x.astype(F32)
    over = np.abs(y.astype(np.float64)) > np.abs(x)
    return np.where(over, np.nextafter(y, F32(0)), y)


def r_rmm(x):
    m, e = np.frexp(x)
    s = m * 2.0 ** 24
    t = np.trunc(s + np.copysign(0.5, s))
    return np.ldexp(t, e - 24).astype(F32)


def r_bits(p):
    def f(x):
        m, e = np.frexp(x)
        return np.ldexp(np.rint(m * 2.0 ** p), e - p).astype(F32)
    return f


# ─────────────────────────── implementation models ───────────────────────────

def products(A, B, k):
    return np.outer(A[:, k], B[:, k])          # exact in binary64


def seq(A, B, order, rnd=r_rne):
    acc = np.zeros((A.shape[0], B.shape[0]), F32)
    for k in order:
        acc = rnd(acc.astype(np.float64) + products(A, B, k))
    return acc


def grouped(A, B, S, rnd=r_rne):
    K = A.shape[1]
    acc = np.zeros((A.shape[0], B.shape[0]), F32)
    for k0 in range(0, K, S):
        blk = sum(products(A, B, k) for k in range(k0, min(k0 + S, K)))
        acc = rnd(acc.astype(np.float64) + blk)
    return acc


def pairwise(A, B):
    terms = [products(A, B, k).astype(F32) for k in range(A.shape[1])]
    while len(terms) > 1:
        nxt = [r_rne(terms[i].astype(np.float64) + terms[i + 1]) for i in range(0, len(terms) - 1, 2)]
        if len(terms) % 2:
            nxt.append(terms[-1])
        terms = nxt
    # C starts at zero: one more exact addition
    return r_rne(np.zeros_like(terms[0], np.float64) + terms[0])


def words(C):
    return [int(w) for w in np.ascontiguousarray(C, F32).view(np.uint32).ravel()]


def models(A, B, K):
    order = list(range(K))
    return {
        "sequential": (seq(A, B, order), K),
        "reversed": (seq(A, B, order[::-1]), K),
        "groups of 16": (grouped(A, B, 16), -(-K // 16)),
        "groups of 32": (grouped(A, B, 32), -(-K // 32)),
        "pairwise tree": (pairwise(A, B), max(1, (K - 1).bit_length() + 1)),
    }


# ─────────────────────────── tests ───────────────────────────

FAILURES = []


def expect(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILURES.append(msg)


def test_regen():
    print("[reproducibility]")
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, str(REPO / "generator" / "gen_vectors.py"), "--out", tmp],
                       check=True, stdout=subprocess.DEVNULL)
        for d in sorted((REPO / "cases").iterdir()):
            names = [p.name for p in (Path(tmp) / "cases" / d.name).iterdir()]
            same = all(filecmp.cmp(d / n, Path(tmp) / "cases" / d.name / n, shallow=False) for n in names)
            expect(same, f"{d.name}: {len(names)} generated files identical")
        expect(filecmp.cmp(REPO / "index.json", Path(tmp) / "index.json", shallow=False), "index.json identical")


def test_case(d):
    case = check.Case(d)
    A, B = load_ab(case)
    K = case.K
    print(f"[{d.name}]  {case.meta['criterion']['type']}")
    ms = models(A, B, K)
    ref_bits = None
    for name, (C, G) in ms.items():
        w = words(C)
        r = check.evaluate(case, w)
        if not case.bitexact:
            rt = check.evaluate(case, w, rounds=G)
            expect(r["passed"] and rt["passed"],
                   f"{name:14s} passes (G=K: {r['worst_ratio']:.2e} of the bound; "
                   f"own G={G}: {rt['worst_ratio']:.2e})")
        else:
            if ref_bits is None:
                ref_bits = w
            expect(r["passed"] and w == ref_bits, f"{name:14s} passes, bit-identical to the others")

    # Faults. The +1% element sits at the output with the least cancellation, where the
    # bound is tightest relative to the value (the most favourable place to detect it).
    C0 = ms["sequential"][0]
    cstar, absum = np.array(case.cstar), np.array(case.absum)
    kappa = np.full(cstar.shape, np.inf)
    nz = cstar != 0
    kappa[nz] = absum[nz] / np.abs(cstar[nz])
    one = C0.copy().ravel()
    i1 = int(np.argmin(kappa))
    one[i1] = F32(float(one[i1]) * 1.01)
    faults = {"one element +1%": one.reshape(C0.shape),
              "accumulator not cleared": (C0.astype(np.float64) * 2).astype(F32)}
    if case.meta["tile"]["instructions"] >= 2:
        faults["last instruction missing"] = seq(A[:, :K - 32], B[:, :K - 32], list(range(K - 32)))
    if case.M == case.N:
        faults["A and B swapped"] = C0.T.copy()
    # Precision-level faults are only visible where rounding or the 24th bit matter.
    if "rounding_coverage" in case.meta:
        faults["truncating accumulator"] = seq(A, B, list(range(K)), r_rtz)
        faults["ties away from zero"] = seq(A, B, list(range(K)), r_rmm)
    if case.bitexact:
        faults["23-bit accumulator"] = seq(A, B, list(range(K)), r_bits(23))
    for name, C in faults.items():
        r = check.evaluate(case, words(C))
        expect(not r["passed"], f"fault '{name}' detected ({r['failures']} elements)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-regen", action="store_true")
    ap.add_argument("--only")
    args = ap.parse_args()
    if not args.skip_regen and not args.only:
        test_regen()
    for d in sorted((REPO / "cases").iterdir()):
        if args.only and not d.name.startswith(args.only):
            continue
        test_case(d)
    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())

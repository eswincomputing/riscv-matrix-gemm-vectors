#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Beijing ESWIN Computing Technology Co., Ltd.
"""Check a dumped C region against the reference of one case.

    python3 check.py 001 c_dump.hex                # 32-bit words, one per line ($writememh)
    python3 check.py 001 c_dump.bin                # raw little-endian bytes
    python3 check.py 004 c_dump.hex --group 16     # implementation rounds once per 16 products
    python3 check.py 004 c_dump.hex --rounds 8     # ... or give the rounding count G directly
    python3 check.py 001 --mailbox mbox.bin        # flags written by the program itself

Only the Python standard library is needed (Python 3.8 or later). Exit status: 0 pass,
1 fail.

Error-bound cases (001-008) pass when every element satisfies

    |C_hat - C*| <= gamma(G) * sum|a*b|,   gamma(G) = G*u / (1 - G*u)

with C* the exact sum of the products. The default G is K, which every implementation
that accumulates in FP32 must meet whatever its order of additions. --group S or
--rounds G gives the rounding count of a specific implementation for a tighter check;
--acc-bits p sets u = 2^-p for an accumulator other than FP32.

Bit-exact cases (009-010) pass only when every element matches the expected FP32 bits.
Their data are built so that any compliant implementation produces exactly these bits.
On failure the script compares the result with other rounding behaviours and with
common data-path faults, and reports the closest match.
"""

import argparse
import json
import math
import re
import struct
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
C_PRESET = 0xA5A5A5A5


# ─────────────────────────── input ───────────────────────────

def load_words(path):
    """Read a .hex file (one 32-bit hex word per line, // and /* */ comments and one
    leading @address line allowed) or raw little-endian bytes. The format is decided by
    whether the whole file parses as hex text, not by the file extension."""
    data = Path(path).read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = None
    if text is not None:
        body = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
        body = re.sub(r"//[^\n]*", " ", body)
        toks = body.split()
        if toks and re.fullmatch(r"@[0-9a-fA-F_]+", toks[0]):
            toks = toks[1:]          # start address written by some $writememh versions
        if toks and all(re.fullmatch(r"[0-9a-fA-F_]{1,8}", t) for t in toks):
            return [int(t.replace("_", ""), 16) for t in toks]
    if len(data) % 4:
        sys.exit(f"{path}: not hex text, and its length {len(data)} is not a multiple of 4")
    return list(struct.unpack(f"<{len(data) // 4}I", data))


def read_f64(path):
    data = Path(path).read_bytes()
    return list(struct.unpack(f"<{len(data) // 8}d", data))


def read_u32(path):
    data = Path(path).read_bytes()
    return list(struct.unpack(f"<{len(data) // 4}I", data))


def f32(u):
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


def f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def ulp_distance(a, b):
    """Distance in FP32 ULPs between two bit patterns; None if the signs differ."""
    if a == b:
        return 0
    if (a >> 31) != (b >> 31):
        return None if (a | b) & 0x7FFFFFFF else 0
    return abs((a & 0x7FFFFFFF) - (b & 0x7FFFFFFF))


def round_to_f32(x, mode="rne", precision=24):
    """Correctly round an exact value (Fraction) to binary32 bits, normal range only.
    Same definition as in generator/gen_vectors.py."""
    if x == 0:
        return 0x80000000 if mode == "rdn" else 0
    neg = x < 0
    a = -x if neg else x
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    if Fraction(2) ** (e + 1) <= a:
        e += 1
    if not -126 <= e <= 127:
        return None
    scaled = a / Fraction(2) ** (e - (precision - 1))
    q, r = divmod(scaled.numerator, scaled.denominator)
    rem = Fraction(r, scaled.denominator)
    half = Fraction(1, 2)
    up = {"rne": rem > half or (rem == half and q & 1), "rmm": rem >= half, "rtz": False,
          "rup": rem > 0 and not neg, "rdn": rem > 0 and neg}[mode]
    v = Fraction(q + int(up)) * Fraction(2) ** (e - (precision - 1))
    return f32_bits(float(-v if neg else v))


def gamma(G, u):
    return G * u / (1.0 - G * u)


# ─────────────────────────── case ───────────────────────────

def find_case(cid):
    hits = sorted((HERE / "cases").glob(f"{cid}*"))
    if not hits:
        names = ", ".join(p.name for p in sorted((HERE / "cases").iterdir()))
        sys.exit(f"no case '{cid}'; available: {names}")
    return hits[0]


class Case:
    def __init__(self, path):
        self.dir = Path(path)
        self.meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
        s = self.meta["shape"]
        self.M, self.N, self.K = s["M"], s["N"], s["K"]
        self.n = self.M * self.N
        self.cstar = read_f64(self.dir / "c_exact.bin")
        self.absum = read_f64(self.dir / "c_absum.bin")
        self.bitexact = self.meta["criterion"]["type"] == "bit-exact"
        self.expect = read_u32(self.dir / "c_expect.bin") if self.bitexact else None


# ─────────────────────────── diagnosis ───────────────────────────

def layout_hints(case, got, matches):
    """Translate common failure patterns into likely causes. matches[i] is True when
    element i is acceptable."""
    M, N, n = case.M, case.N, case.n
    out = []
    preset = sum(1 for w in got if w == C_PRESET)
    if preset == n:
        return ["Every element still holds the image preset 0xA5A5A5A5: the result store "
                "did not run or wrote elsewhere. Check that the done flag is set."]
    if preset:
        out.append(f"{preset} element(s) still hold the preset 0xA5A5A5A5: part of C was not "
                   "stored. Missing trailing rows or columns point at the tile size settings; "
                   "scattered elements point at the store stride.")
    if all((w & 0x7FFFFFFF) == 0 for w in got):
        return out + ["Every element is zero: the accumulator was cleared but nothing was "
                      "accumulated. Check that the multiply-accumulate instructions executed."]
    rows_ok = [all(matches[m * N + j] for j in range(N)) for m in range(M)]
    cols_ok = [all(matches[i * N + nn] for i in range(M)) for nn in range(N)]
    good_rows = sum(rows_ok)
    if M > 1 and good_rows == 1 and rows_ok[0]:
        out.append("Only row 0 is correct: the row count of the tile (mtilem) did not take "
                   "effect, or only one row was computed.")
    elif 0 < good_rows < M:
        first_half = all(rows_ok[:M // 2]) and not any(rows_ok[M // 2:])
        out.append(f"{good_rows}/{M} rows are fully correct"
                   + (" (exactly the first half: check that clearing the accumulator covers "
                      "all rows)." if first_half else "."))
    good_cols = sum(cols_ok)
    if 0 < good_cols < N:
        out.append(f"{good_cols}/{N} columns are fully correct: check the column count "
                   "(mtilen) and the store stride.")
    if M == N:
        ok_t = 0
        for m in range(M):
            for nn in range(N):
                ok_t += element_ok(case, got[m * N + nn], nn * N + m)
        if ok_t > 0.9 * n and sum(matches) < 0.9 * n:
            out.append("The result matches the transpose of the reference: the A and B "
                       "operands are swapped in the multiply-accumulate instruction.")
    return out


def element_ok(case, word, i, tol=None):
    if case.bitexact:
        return word == case.expect[i]
    x = f32(word)
    if math.isnan(x) or math.isinf(x):
        return False
    t = tol if tol is not None else gamma(case.K, 2.0 ** -24)
    return abs(x - case.cstar[i]) <= t * case.absum[i]


def rounding_hints(case, got, bad):
    """For bit-exact cases: which rounding behaviour explains the mismatches."""
    models = [("round toward zero (truncation)", "rtz", 24),
              ("round to nearest, ties away from zero", "rmm", 24),
              ("round toward +infinity", "rup", 24), ("round toward -infinity", "rdn", 24),
              ("round to nearest even with a 23-bit significand", "rne", 23),
              ("round to nearest even with a 22-bit significand", "rne", 22)]
    sample = bad[:20000]
    best = []
    for label, mode, p in models:
        hit = 0
        for i in sample:
            r = round_to_f32(Fraction(case.cstar[i]), mode, p)
            hit += r == got[i]
        best.append((hit, label))
    best.sort(reverse=True)
    hit, label = best[0]
    if hit >= 0.9 * len(sample):
        return [f"{hit} of {len(sample)} mismatching elements equal the exact sum rounded with "
                f"'{label}'. Expected: round to nearest, ties to even, 24-bit significand."]
    return []


# ─────────────────────────── evaluation ───────────────────────────

def evaluate(case, got, rounds=None, u=2.0 ** -24):
    """Return a dict describing the verdict for the words in got (at least M*N)."""
    got = got[:case.n]
    res = {"case": case.meta["id"], "criterion": case.meta["criterion"]["type"], "n": case.n}
    if case.bitexact:
        bad = [i for i in range(case.n) if got[i] != case.expect[i]]
        res.update(passed=not bad, failures=len(bad), first=bad[:10])
        if bad:
            matches = [got[i] == case.expect[i] for i in range(case.n)]
            res["hints"] = rounding_hints(case, got, bad) or layout_hints(case, got, matches)
        return res

    G = rounds if rounds is not None else case.meta["criterion"]["default_rounds_G"]
    tol = gamma(G, u)
    worst_ratio, need_G, bad = 0.0, 0.0, []
    max_ulp, n_exact = 0, 0
    for i in range(case.n):
        x = f32(got[i])
        c, s = case.cstar[i], case.absum[i]
        if math.isnan(x) or math.isinf(x):
            bad.append(i)
            continue
        err = abs(x - c)
        if s > 0:
            worst_ratio = max(worst_ratio, err / (tol * s))
            need_G = max(need_G, err / (u * s))
        elif err > 0:
            bad.append(i)
            continue
        if err > tol * s:
            bad.append(i)
        cr = f32_bits(c) if c == c else None
        d = ulp_distance(got[i], cr) if cr is not None else None
        if d == 0:
            n_exact += 1
        elif d is not None:
            max_ulp = max(max_ulp, d)
    res.update(passed=not bad, failures=len(bad), first=bad[:10], rounds=G, u=u, tol=tol,
               worst_ratio=worst_ratio, min_rounds=math.ceil(need_G - 1e-9) if need_G else 0,
               correctly_rounded=n_exact, max_ulp_from_correctly_rounded=max_ulp)
    if bad:
        matches = [element_ok(case, got[i], i, tol) for i in range(case.n)]
        res["hints"] = layout_hints(case, got, matches)
    return res


def report(case, res, got, verbose=False):
    meta = case.meta
    print(f"case        {meta['id']} {meta['name']}  ({meta['title']})")
    print(f"shape       M={case.M} N={case.N} K={case.K}, {meta['tile']['instructions']} instruction(s)")
    print(f"criterion   {res['criterion']}")
    if not case.bitexact:
        print(f"            |C_hat - C*| <= gamma(G) * sum|a*b|   G = {res['rounds']}, "
              f"u = 2^{math.log2(res['u']):.0f}, gamma = {res['tol']:.3e}")
        print(f"worst       {res['worst_ratio']:.3e} of the bound"
              f"   (these errors would still pass with G as small as {res['min_rounds']};"
              f" this is not the implementation's rounding count)")
        print(f"for info    {res['correctly_rounded']}/{case.n} elements equal the correctly "
              f"rounded exact sum; the largest distance from it is "
              f"{res['max_ulp_from_correctly_rounded']} ULP")
        cr = meta.get("cancellation_ratio") or {}
        if cr.get("max"):
            print(f"            cancellation ratio sum|a*b| / |C*|: median {cr['median']}, "
                  f"max {cr['max']} - ULP counts are not a usable criterion here")
    if res["passed"]:
        print("\n>>> PASS")
        return
    print(f"\n>>> FAIL   {res['failures']} of {case.n} elements")
    print("    index   (m, n)        got          expected")
    for i in res["first"][:10 if not verbose else 50]:
        m, n = divmod(i, case.N)
        exp = (f"0x{case.expect[i]:08x}" if case.bitexact else f"{case.cstar[i]:.9g} (exact)")
        print(f"   {i:6d}  ({m:3d},{n:3d})  {f32(got[i]):.9g} [0x{got[i]:08x}]  {exp}")
    hints = res.get("hints") or []
    if hints:
        print("\nlikely causes:")
        for h in hints:
            print(f"  * {h}")


MAILBOX_FIELDS = ["done", "status", "fail_index", "fail_count", "tol_lo", "tol_hi",
                  "trap_cause", "trap_epc", "trap_tval"]


def check_mailbox(case, words):
    """Verdict from the mailbox the program writes (36 bytes, see meta.json "mailbox")."""
    n = len(MAILBOX_FIELDS)
    if len(words) < n:
        sys.exit(f"the mailbox dump has {len(words)} words; {n} are needed "
                 f"({4 * n} bytes from {case.meta['mailbox']['done']})")
    mb = dict(zip(MAILBOX_FIELDS, words[:n]))
    mk = case.meta["markers"]
    done = mb["done"] == int(mk["done"], 16)
    passed = done and mb["status"] == int(mk["pass"], 16)
    status = {int(mk["pass"], 16): "PASS", int(mk["fail"], 16): "FAIL"}.get(mb["status"], "not written")
    lines = [f"done        0x{mb['done']:08x}  ({'set' if done else 'NOT set: the program did not finish'})",
             f"status      0x{mb['status']:08x}  ({status})"]
    if not case.bitexact:
        tol = struct.unpack("<d", struct.pack("<II", mb["tol_lo"], mb["tol_hi"]))[0]
        G = tol / (2.0 ** -24 * (1.0 + tol)) if tol > 0 else 0.0
        lines.append(f"tolerance   {tol:.6e}  (gamma(G) with G = {G:.0f})")
    if mb["trap_cause"]:
        lines.append(f"exception   mcause 0x{mb['trap_cause']:08x}  mepc 0x{mb['trap_epc']:08x}  "
                     f"mtval 0x{mb['trap_tval']:08x}")
        passed = False
    if mb["fail_count"]:
        first = mb["fail_index"] - (1 << 32) if mb["fail_index"] >> 31 else mb["fail_index"]
        lines.append(f"self-check  {mb['fail_count']} element(s) failed, first index {first}")
    return passed, lines


def main():
    ap = argparse.ArgumentParser(description="Check a dumped C region against the reference.")
    ap.add_argument("case", help="case id or directory name, e.g. 001 or 001_full_tile")
    ap.add_argument("dump", nargs="?", help="dumped C region (.hex text or raw .bin)")
    ap.add_argument("--mailbox", help="dumped 36-byte mailbox (.hex text or raw .bin)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--rounds", type=int, help="rounding count G of the implementation")
    g.add_argument("--group", type=int, help="the implementation rounds once per GROUP products")
    ap.add_argument("--acc-bits", type=int, default=24,
                    help="significand bits of the accumulator (u = 2^-bits), default 24")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    if not args.dump and not args.mailbox:
        ap.error("give a C dump, --mailbox, or both")

    case = Case(find_case(args.case))
    ok = True
    if args.mailbox:
        passed, lines = check_mailbox(case, load_words(args.mailbox))
        print(f"case        {case.meta['id']} {case.meta['name']}  ({case.meta['title']})  - mailbox")
        print("\n".join(lines))
        print(f"\n>>> {'PASS' if passed else 'FAIL'} (mailbox)\n")
        ok &= passed
    if args.dump:
        got = load_words(args.dump)
        if len(got) < case.n:
            sys.exit(f"the dump has {len(got)} words; {case.n} are needed "
                     f"({case.n * 4} bytes from {case.meta['segments']['c']})")
        rounds = args.rounds
        if args.group:
            rounds = -(-case.K // args.group)
        res = evaluate(case, got, rounds, 2.0 ** -args.acc_bits)
        report(case, res, got, args.verbose)
        ok &= res["passed"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

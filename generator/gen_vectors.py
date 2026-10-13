#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Beijing ESWIN Computing Technology Co., Ltd.
"""Generate the matrix-extension GEMM test vectors.

For every case this script writes the input matrices, an implementation-independent
reference, a self-checking bare-metal program (prog.S + prog.ld) and meta.json.
Assemble the programs afterwards with generator/build.sh.

    python3 generator/gen_vectors.py                    # all cases into ./cases
    python3 generator/gen_vectors.py --only 010         # one case
    python3 generator/gen_vectors.py --base 0x80000000  # DRAM base of your platform

Reference model
---------------
The operation under test is C = A * B^T with BF16 inputs and FP32 accumulation.
What the arithmetic fixes is that every BF16 x BF16 product is exact, and that the
products are accumulated into an FP32 accumulator. What it leaves to the
implementation is the order of the additions, where rounding happens and how many
times. The reference therefore contains only the fixed part:

    C*[m][n]      = sum_k A[m][k] * B[n][k]       exact sum of the products
    absum[m][n]   = sum_k |A[m][k] * B[n][k]|     magnitude scale, order independent

Both are computed with math.fsum (a correctly rounded sum) and stored as binary64.
The rounding error of that final conversion is at most 2^-53 * absum, many orders
of magnitude below any tolerance used here. Every product of two BF16 values has at
most 16 significant bits and is therefore exact in binary64.

Implementation-defined behaviour enters only the acceptance criterion:

    |C_hat - C*| <= gamma(G) * absum,   gamma(G) = G*u / (1 - G*u),   u = 2^-24

where G is the number of FP32 roundings an output goes through. G = K (one rounding
per product, any order) is valid for every implementation that accumulates in FP32;
an implementation that rounds once per group of S products may use G = ceil(K/S).
See docs/METHOD.md for the derivation.

Cases 009 and 010 are built so that the FP32 result does not depend on the order of
accumulation; for them the criterion is bit-exact.
"""

import argparse
import json
import math
import struct
import textwrap
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
U32 = 2.0 ** -24                    # unit roundoff of binary32, round-to-nearest
K_INSN = 32                         # BF16 elements of K covered by one multiply-accumulate


def gamma(G: int) -> float:
    """gamma(G) = G*u / (1 - G*u): rigorous form of the G*u bound."""
    return G * U32 / (1.0 - G * U32)


# ═══════════════════════════ number formats ═══════════════════════════

def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def bits_f32(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


def f64_bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def bf16_value(u16: int) -> float:
    """BF16 bit pattern -> exact value (BF16 is the upper half of binary32)."""
    return bits_f32((u16 & 0xFFFF) << 16)


def to_bf16_rne(x: float) -> int:
    """Value -> BF16 bit pattern: to binary32 first, then round-to-nearest-even to BF16."""
    u = f32_bits(x)
    if (u & 0x7F800000) == 0x7F800000:
        return (u >> 16) & 0xFFFF
    u += 0x7FFF + ((u >> 16) & 1)
    return (u >> 16) & 0xFFFF


def round_to_f32(x: Fraction, mode: str = "rne", precision: int = 24) -> int:
    """Correctly round an exact value to binary32 (normal range only); returns bits.

    mode: rne (nearest, ties to even), rmm (nearest, ties away), rtz, rdn, rup.
    precision < 24 emulates an accumulator with fewer significant bits; the result is
    still a binary32 value. A zero result is +0, except -0 for rdn (IEEE 754 sign
    rule for an exact zero sum of operands with opposite signs).
    """
    if x == 0:
        return 0x80000000 if mode == "rdn" else 0
    neg = x < 0
    a = -x if neg else x
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    if Fraction(2) ** (e + 1) <= a:
        e += 1
    assert -126 <= e <= 127, "outside the normal binary32 range"
    scaled = a / (Fraction(2) ** (e - (precision - 1)))
    q, r = divmod(scaled.numerator, scaled.denominator)
    rem = Fraction(r, scaled.denominator)
    half = Fraction(1, 2)
    if mode == "rne":
        up = rem > half or (rem == half and q & 1)
    elif mode == "rmm":
        up = rem >= half
    elif mode == "rtz":
        up = False
    elif mode == "rup":
        up = rem > 0 and not neg
    elif mode == "rdn":
        up = rem > 0 and neg
    else:
        raise ValueError(mode)
    q += int(up)
    v = Fraction(q) * Fraction(2) ** (e - (precision - 1))
    return f32_bits(float(-v if neg else v))


# ═══════════════════════════ deterministic data ═══════════════════════════

class Rng:
    """xorshift64: deterministic on every platform and Python version."""

    def __init__(self, seed: int):
        self.s = seed & 0xFFFFFFFFFFFFFFFF or 0x9E3779B97F4A7C15

    def next(self) -> int:
        s = self.s
        s ^= (s << 13) & 0xFFFFFFFFFFFFFFFF
        s ^= s >> 7
        s ^= (s << 17) & 0xFFFFFFFFFFFFFFFF
        self.s = s
        return s

    def uniform(self) -> float:
        """Uniform on [-1, 1)."""
        return (self.next() >> 11) / float(1 << 53) * 2.0 - 1.0

    def uniform_case008(self) -> float:
        """Uniform on [-1, 3). Used only by case 008, whose data are kept fixed (see CASES)."""
        return (self.next() >> 11) / float(1 << 52) * 2.0 - 1.0

    def integer(self, lo: int, hi: int) -> int:
        """Integer in [lo, hi]."""
        return lo + self.next() % (hi - lo + 1)


def gen_zero_mean(rows, cols, rng):
    """Zero-mean data whose magnitude depends on (i, j): a transposed or shifted
    access pattern changes the result even when every element is read."""
    return [[to_bf16_rne(rng.uniform() * (1.0 + 0.5 * ((i * 7 + j * 3) % 5)))
             for j in range(cols)] for i in range(rows)]


def gen_case008(rows, cols, rng):
    """About 1% of the entries are +-6912, the rest uniform on [-1, 3). Large outliers
    in a few channels are typical of LLM activations; together with the sign mix they
    produce partial sums that are orders of magnitude larger than many final results."""
    out = []
    for i in range(rows):
        row = []
        for j in range(cols):
            v = rng.uniform_case008()
            if (i * 13 + j * 29) % 97 == 0:
                v = 6912.0 * (1.0 if (i + j) % 2 == 0 else -1.0)
            row.append(to_bf16_rne(v))
        out.append(row)
    return out


# ═══════════════════════════ reference ═══════════════════════════

def reference(A, B, M, N, K):
    """Return (C*, absum) as flat lists of floats; A is [M][K], B is [N][K] BF16 bits."""
    Av = [[bf16_value(u) for u in row[:K]] for row in A]
    Bv = [[bf16_value(u) for u in row[:K]] for row in B]
    cstar, absum = [], []
    for m in range(M):
        a = Av[m]
        for n in range(N):
            p = [x * y for x, y in zip(a, Bv[n])]          # each product is exact
            cstar.append(math.fsum(p))
            absum.append(math.fsum(abs(t) for t in p))
    return cstar, absum


def exact_sums(A, B, M, N, K):
    """Exact sums as Fractions (used for the bit-exact cases)."""
    out = []
    for m in range(M):
        for n in range(N):
            s = Fraction(0)
            for k in range(K):
                if A[m][k] & 0x7FFF and B[n][k] & 0x7FFF:
                    s += Fraction(bf16_value(A[m][k])) * Fraction(bf16_value(B[n][k]))
            out.append(s)
    return out


def cancellation(cstar, absum, M, N, top=8):
    """kappa = absum / |C*|: how much the result shrank through cancellation."""
    ks = sorted(((s / abs(c), i) for i, (c, s) in enumerate(zip(cstar, absum)) if c != 0),
                reverse=True)
    if not ks:
        return {"median": None, "max": None, "top": []}
    return {
        "median": round(ks[len(ks) // 2][0], 3),
        "max": round(ks[0][0], 1),
        "top": [{"m": i // N, "n": i % N, "kappa": round(k, 1)} for k, i in ks[:top]],
    }


# ═══════════════════════════ memory layout ═══════════════════════════
#
# One address map for every case, so a testbench is written once.
# Offsets are relative to --base. Everything the program needs is inside the image.

SEG = {
    "text":   0x000000,   # code
    "mbox":   0x00F000,   # mailbox: flags, tolerance and exception record (36 bytes)
    "a":      0x010000,   # A, BF16, <= 256 KiB
    "b":      0x050000,   # B, BF16, <= 256 KiB
    "c":      0x090000,   # C, FP32, <= 64 KiB, preset to C_PRESET
    "expect": 0x0A0000,   # self-check data: C* (FP64) or the exact FP32 result
    "absum":  0x0C0000,   # sum|a*b| (FP64), error-bound cases only
}
SEG_SIZE = {"text": 0xF000, "a": 0x40000, "b": 0x40000, "c": 0x10000,
            "expect": 0x20000, "absum": 0x20000}
MBOX = {
    "done": 0x00, "status": 0x04, "fail_index": 0x08, "fail_count": 0x0C,
    "tol": 0x10,                 # FP64 gamma(G) used by the self-check (8 bytes)
    "trap_cause": 0x18, "trap_epc": 0x1C, "trap_tval": 0x20,
}
MBOX_BYTES = 0x24
DONE_MAGIC = 0x600D600D
ST_PASS = 0x50415353            # "PASS"
ST_FAIL = 0x4641494C            # "FAIL"
C_PRESET = 0xA5A5A5A5           # a missed store leaves this, not a plausible 0
CSR = {"mtilem": 0x803, "mtilen": 0x804, "mtilek": 0x805}


# ═══════════════════════════ cases ═══════════════════════════

class Case:
    def __init__(self, cid, name, M, N, K, title, purpose, kind="bound", data="zero_mean",
                 pad_a=0, pad_b=0):
        self.cid, self.name, self.M, self.N, self.K = cid, name, M, N, K
        self.title, self.purpose = title, purpose
        self.kind = kind            # bound | exact_int | single_round
        self.data = data
        self.pad_a, self.pad_b = pad_a, pad_b
        self.seed = 0xA3E00000 + int(cid)

    @property
    def bitexact(self):
        return self.kind != "bound"

    @property
    def dirname(self):
        return f"{self.cid}_{self.name}"

    @property
    def n_insn(self):
        return (self.K + K_INSN - 1) // K_INSN


CASES = [
    Case("001", "full_tile", 128, 128, 32, "Full tile",
         "All of M, N and K at their maximum for one instruction. Exercises every "
         "output of the tile. Run this case first; a failure here is fundamental."),
    Case("002", "m1_gemv", 1, 128, 1024, "M = 1 (matrix-vector)",
         "The shape of token-by-token LLM decoding: one row of A, 32 accumulating "
         "instructions. Only 1/128 of the tile rows are used."),
    Case("003", "partial_m", 7, 128, 32, "Partial M",
         "M = 7, not a power of two, so implicit power-of-two assumptions fail."),
    Case("004", "k_multiblock", 128, 128, 256, "Multi-instruction accumulation",
         "K = 256: eight instructions accumulate into the same accumulator, so the "
         "accumulator is carried across instructions."),
    Case("005", "partial_n", 128, 13, 32, "Partial N",
         "N = 13 (prime). The column dimension of a partial tile takes a different "
         "path from the row dimension and is tested separately."),
    Case("006", "partial_k", 128, 128, 17, "Partial K",
         "K = 17: the unused product lanes must contribute zero, not stale values. "
         "The row stride is padded to 18 elements for 4-byte alignment.",
         pad_a=1, pad_b=1),
    Case("007", "stride_gap", 128, 128, 32, "Row stride larger than the row",
         "A and B rows are 16 elements longer than K. Loads must follow the stride; "
         "an implementation that assumes packed rows fails only here.",
         pad_a=16, pad_b=16),
    Case("008", "outlier_activations", 128, 128, 1024, "Outliers and cancellation",
         "About 1% of the entries are +-6912; together with mixed signs this produces "
         "partial sums far larger than many of the final results (cancellation ratio "
         "up to about 1e5). Its data are fixed and do not follow changes to the other "
         "generators, so that results reported for this case stay reproducible.",
         data="case008"),
    Case("009", "exact_integer", 128, 128, 1024, "Integer data, exact in any order",
         "All products are integers and sum|a*b| < 2^24, so every partial sum in any "
         "order is exactly representable in FP32. A quarter of the outputs lie in "
         "[2^23, 2^24); the odd ones among them need all 24 significant bits. The "
         "result is bit-exact for every compliant implementation; a narrower "
         "accumulator or any data-path error shows up as a bit mismatch.",
         kind="exact_int", data="int"),
    Case("010", "single_rounding", 128, 128, 64, "One rounding per output",
         "Each output has exactly two non-zero products, in different instructions, "
         "so exactly one addition can be inexact whatever the order. The pairs cover "
         "exact 24-bit sums, ties with even and odd neighbours, values just above and "
         "below the halfway point, far-below (sticky) bits and exact cancellation. "
         "Expected values are round-to-nearest-even; the checker names the rounding "
         "mode that matches a failing result.",
         kind="single_round", data="special"),
]


# ═══════════════════════════ special data ═══════════════════════════

def gen_exact_int(c, rng):
    """Case 009. Rows 0..63 of A and B are positive (large sums), rows 64..127 are
    zero-mean (cancellation). Bounds keep sum|a*b| < 2^24 for every output."""
    h = c.M // 2
    A = [[to_bf16_rne(float(rng.integer(128, 255) if i < h else rng.integer(-15, 15)))
          for _ in range(c.K)] for i in range(c.M)]
    B = [[to_bf16_rne(float(rng.integer(32, 64) if j < c.N // 2 else rng.integer(-15, 15)))
          for _ in range(c.K)] for j in range(c.N)]
    return A, B


# q classes of case 010: second product relative to the first product's leading bit.
Q_CLASSES = [
    ("exact_24bit", Fraction(1, 2 ** 23)),
    ("tie", Fraction(1, 2 ** 24)),
    ("tie_odd", Fraction(3, 2 ** 24)),
    ("above_half", Fraction(1, 2 ** 24) * (1 + Fraction(1, 2 ** 7))),
    ("below_half", Fraction(1, 2 ** 25) * (2 - Fraction(1, 2 ** 7))),
    ("sticky", Fraction(1, 2 ** 40)),
    ("equal_magnitude", Fraction(1)),
]
K13 = (5, 45)          # positions of the two non-zero products (different instructions)


def gen_single_round(c, rng):
    """Case 010. Output (m, n) = x_m * s_n + y_m * t_n with
         x_m = +-2^e_m * (1.f_m)     s_n = +-2^sig_n
         y_m = +-2^e_m               t_n = +-2^sig_n * q(n)
       so the second product is q(n) times the first product's leading power of two,
       whatever the exponents. Rows vary significand, exponent and signs; columns
       cycle through the q classes with both signs."""
    A = [[0] * c.K for _ in range(c.M)]
    B = [[0] * c.K for _ in range(c.N)]
    for m in range(c.M):
        e = rng.integer(-12, 12)
        frac = 0 if m % 8 == 0 else rng.integer(1, 127)     # every 8th row: power of two
        sx = -1 if rng.next() & 1 else 1
        sy = -1 if rng.next() & 1 else 1
        x = sx * (1 + frac / 128.0) * 2.0 ** e
        A[m][K13[0]] = to_bf16_rne(x)
        A[m][K13[1]] = to_bf16_rne(sy * 2.0 ** e)
    for n in range(c.N):
        sig = rng.integer(-12, 12)
        _, q = Q_CLASSES[n % len(Q_CLASSES)]
        sq = -1 if (n // len(Q_CLASSES)) % 2 else 1
        ss = -1 if rng.next() & 1 else 1
        B[n][K13[0]] = to_bf16_rne(ss * 2.0 ** sig)
        t = sq * float(q) * 2.0 ** sig
        B[n][K13[1]] = to_bf16_rne(t)
        assert bf16_value(B[n][K13[1]]) == t, "q must be exactly representable in BF16"
    return A, B


def single_round_stats(c, exact):
    """How many outputs each rounding mode / narrow accumulator would get wrong."""
    rne = [round_to_f32(x, "rne") for x in exact]
    stats = {"inexact": sum(1 for x, b in zip(exact, rne) if Fraction(bits_f32(b)) != x)}
    for mode in ("rtz", "rmm", "rup", "rdn"):
        stats[f"differs_under_{mode}"] = sum(1 for x, b in zip(exact, rne)
                                             if round_to_f32(x, mode) != b)
    stats["differs_with_23_bit_accumulator"] = sum(
        1 for x, b in zip(exact, rne) if round_to_f32(x, "rne", 23) != b)
    stats["exact_zero"] = sum(1 for x in exact if x == 0)
    per_class = {}
    for i, x in enumerate(exact):
        name = Q_CLASSES[(i % c.N) % len(Q_CLASSES)][0]
        sign = "-" if ((i % c.N) // len(Q_CLASSES)) % 2 else "+"
        per_class[f"{name}{sign}"] = per_class.get(f"{name}{sign}", 0) + 1
    stats["outputs_per_class"] = per_class
    return rne, stats


# ═══════════════════════════ file writers ═══════════════════════════

def bf16_words(Mtx, stride_elems):
    """Rows of BF16 -> little-endian 32-bit words (low half = even index)."""
    assert stride_elems % 2 == 0
    words = []
    for row in Mtx:
        r = list(row) + [0] * (stride_elems - len(row))
        words += [r[i] | (r[i + 1] << 16) for i in range(0, stride_elems, 2)]
    return words


def write_u32(path, words):
    path.write_bytes(b"".join(struct.pack("<I", w & 0xFFFFFFFF) for w in words))


def write_f64(path, values):
    path.write_bytes(b"".join(struct.pack("<d", v) for v in values))


# ═══════════════════════════ assembly ═══════════════════════════

def header(c, base):
    L = [f"# Generated by generator/gen_vectors.py - do not edit.",
         f"# Case {c.cid} {c.name}: {c.title}"]
    L += [f"#   {line}" for line in textwrap.wrap(c.purpose, 86, break_on_hyphens=False)]
    L += [f"#", f"# C[{c.M}][{c.N}] = A[{c.M}][{c.K}] * B[{c.N}][{c.K}]^T   (BF16 x BF16 -> FP32)", ""]
    for k, v in SEG.items():
        L.append(f"    .equ {k.upper()}_ADDR, 0x{base + v:08X}")
    for k, v in MBOX.items():
        L.append(f"    .equ MB_{k.upper()}, 0x{base + SEG['mbox'] + v:08X}")
    L += [f"    .equ DONE_MAGIC, 0x{DONE_MAGIC:08X}", f"    .equ ST_PASS,    0x{ST_PASS:08X}",
          f"    .equ ST_FAIL,    0x{ST_FAIL:08X}", f"    .equ N_ELEM,     {c.M * c.N}", ""]
    return L


def prologue(trap):
    L = ['    .section .text.init, "ax"', "    .global _start", "_start:",
         "    csrr    t0, mhartid           # only hart 0 runs", "    bnez    t0, park",
         "#ifdef WITH_UART",
         "    fence.i                       # discard stale instructions after a soft reset",
         "#endif",
         "    # Enable the FP and matrix context status fields (Initial).",
         "    # Without MS (bits 30:29) every matrix instruction is illegal.",
         "    li      t0, (1 << 13) | (1 << 29)   # FS | MS",
         "    csrs    mstatus, t0",
         "    csrw    frm, x0               # round to nearest, ties to even"]
    if trap:
        L += ["    la      t0, trap_entry        # record any exception in the mailbox",
              "    csrw    mtvec, t0"]
    return L + [""]


def matrix_release():
    """Leave matrix use by marking the matrix state Clean (MS = 10)."""
    return ["    li      t0, (3 << 29)", "    csrc    mstatus, t0",
            "    li      t0, (2 << 29)", "    csrs    mstatus, t0"]


def gemm(c, stride_a, stride_b):
    L = ["    # ---- GEMM ----",
         f"    li      t0, {c.M}", f"    csrw    0x{CSR['mtilem']:03x}, t0      # mtilem",
         f"    li      t0, {c.N}", f"    csrw    0x{CSR['mtilen']:03x}, t0      # mtilen",
         "    mzero   acc0"]
    for i in range(c.n_insn):
        k0 = i * K_INSN
        kk = min(K_INSN, c.K - k0)
        L += [f"    # instruction {i}: k = {k0}..{k0 + kk - 1}",
              f"    li      t0, {kk}", f"    csrw    0x{CSR['mtilek']:03x}, t0      # mtilek",
              f"    li      a0, A_ADDR + {k0 * 2}", f"    li      a1, {stride_a}",
              "    mlae16  tr0, (a0), a1",
              f"    li      a0, B_ADDR + {k0 * 2}", f"    li      a1, {stride_b}",
              "    mlbe16  tr1, (a0), a1",
              "    mfmacc.s.bf16 acc0, tr1, tr0   # acc0 += A * B^T (syntax: md, ms2, ms1)"]
    L += ["    li      a0, C_ADDR", f"    li      a1, {c.N * 4}", "    msce32  acc0, (a0), a1"]
    return L + matrix_release()


def cache_inval_c(c):
    """Only in the UART build: discard any cached copy of C before the CPU reads it."""
    return ["#ifdef WITH_UART",
            "    li      a0, C_ADDR", f"    li      a1, C_ADDR + {c.M * c.N * 4}",
            "1:  cbo.inval (a0)", "    addi    a0, a0, 64", "    bltu    a0, a1, 1b",
            "    fence   rw, rw", "#endif"]


def selfcheck_bound():
    """|C - C*| <= tol * absum per element, in binary64. A NaN result fails."""
    return [
        "    # ---- self-check: |C - C*| <= tol * sum|a*b| (binary64) ----",
        "    li      a0, C_ADDR", "    li      a1, EXPECT_ADDR", "    li      a2, ABSUM_ADDR",
        "    li      a3, N_ELEM", "    li      t1, MB_TOL", "    fld     f3, 0(t1)",
        "    li      a4, 0                 # index", "    li      a5, 0                 # failures",
        "    li      a6, -1                # first failing index",
        "chk_loop:", "    beqz    a3, chk_done", "    flw     f0, 0(a0)",
        "    fcvt.d.s f0, f0", "    fld     f1, 0(a1)", "    fld     f2, 0(a2)",
        "    fsub.d  f0, f0, f1", "    fabs.d  f0, f0", "    fmul.d  f2, f2, f3",
        "    fle.d   t0, f0, f2", "    bnez    t0, chk_next", "    addi    a5, a5, 1",
        "    bgez    a6, chk_next", "    mv      a6, a4",
        "chk_next:", "    addi    a0, a0, 4", "    addi    a1, a1, 8", "    addi    a2, a2, 8",
        "    addi    a4, a4, 1", "    addi    a3, a3, -1", "    j       chk_loop", "chk_done:"]


def selfcheck_bits(n):
    return [
        "    # ---- self-check: bit-exact comparison with the expected FP32 result ----",
        "    li      a0, C_ADDR", "    li      a1, EXPECT_ADDR", f"    li      a3, {n}",
        "    li      a4, 0", "    li      a5, 0", "    li      a6, -1",
        "chk_loop:", "    beqz    a3, chk_done", "    lw      t0, 0(a0)", "    lw      t1, 0(a1)",
        "    beq     t0, t1, chk_next", "    addi    a5, a5, 1", "    bgez    a6, chk_next",
        "    mv      a6, a4",
        "chk_next:", "    addi    a0, a0, 4", "    addi    a1, a1, 4", "    addi    a4, a4, 1",
        "    addi    a3, a3, -1", "    j       chk_loop", "chk_done:"]


def store_result():
    return (["    li      t1, MB_FAIL_INDEX", "    sw      a6, 0(t1)",
             "    li      t1, MB_FAIL_COUNT", "    sw      a5, 0(t1)",
             "    li      t0, ST_PASS", "    beqz    a5, 7f", "    li      t0, ST_FAIL",
             "7:  li      t1, MB_STATUS", "    sw      t0, 0(t1)",
             "    li      t0, DONE_MAGIC", "    li      t1, MB_DONE", "    sw      t0, 0(t1)",
             "#ifdef WITH_UART",
             "    li      a0, MB_DONE           # make the mailbox visible to a host reading DRAM",
             "    cbo.clean (a0)", "    fence   rw, rw", "#endif"])


def uart(c):
    """Optional serial report: build with -DWITH_UART -DUART_BASE=<addr>. The default
    register layout is a byte-wide 16550 (LSR at offset 5); for 32-bit registers on a
    4-byte pitch add -DUART_REG_WORD -DUART_LSR_OFF=0x14."""
    tag = f"case {c.cid} {c.name}  M={c.M} N={c.N} K={c.K}"
    return ["#ifdef WITH_UART",
            "#ifndef UART_BASE", "#error define UART_BASE for the UART build", "#endif",
            "#ifndef UART_LSR_OFF", "#define UART_LSR_OFF 5", "#endif",
            "#ifdef UART_REG_WORD", "#define UART_LD lw", "#define UART_ST sw",
            "#else", "#define UART_LD lbu", "#define UART_ST sb", "#endif",
            "    la      a0, umsg_hdr", "    jal     ra, uputs",
            "    li      t1, MB_STATUS", "    lw      t0, 0(t1)", "    li      t2, ST_PASS",
            "    beq     t0, t2, upass",
            "    la      a0, umsg_fail", "    jal     ra, uputs",
            "    li      t1, MB_FAIL_INDEX", "    lw      a0, 0(t1)", "    jal     ra, uputhex",
            "    la      a0, umsg_cnt", "    jal     ra, uputs",
            "    li      t1, MB_FAIL_COUNT", "    lw      a0, 0(t1)", "    jal     ra, uputhex",
            "    la      a0, umsg_nl", "    jal     ra, uputs", "    j       park",
            "upass:", "    la      a0, umsg_pass", "    jal     ra, uputs", "    j       park",
            "uputs:", "    li      t6, UART_BASE",
            "1:  lbu     t0, 0(a0)", "    beqz    t0, 3f",
            "2:  UART_LD t1, UART_LSR_OFF(t6)", "    andi    t1, t1, 0x20", "    beqz    t1, 2b",
            "    UART_ST t0, 0(t6)", "    addi    a0, a0, 1", "    j       1b", "3:  ret",
            "uputhex:", "    li      t6, UART_BASE", "    li      t2, 28",
            "1:  srl     t0, a0, t2", "    andi    t0, t0, 0xf", "    li      t1, 10",
            "    blt     t0, t1, 2f", "    addi    t0, t0, 'a' - 10", "    j       3f",
            "2:  addi    t0, t0, '0'",
            "3:  UART_LD t1, UART_LSR_OFF(t6)", "    andi    t1, t1, 0x20", "    beqz    t1, 3b",
            "    UART_ST t0, 0(t6)", "    addi    t2, t2, -4", "    bgez    t2, 1b", "    ret",
            '    .section .text.umsg, "a"', "    .align 2",
            f'umsg_hdr:  .asciz "\\r\\n=== {tag} ===\\r\\n"',
            'umsg_pass: .asciz "RESULT: PASS\\r\\n"',
            'umsg_fail: .asciz "RESULT: FAIL  first="',
            'umsg_cnt:  .asciz "  count="', 'umsg_nl:   .asciz "\\r\\n"',
            '    .section .text.init, "ax"', "#endif"]


def trap_handler():
    return ["", "    .align 4", "trap_entry:                       # record the exception and stop",
            "    csrr    t0, mcause", "    li      t1, MB_TRAP_CAUSE", "    sw      t0, 0(t1)",
            "    csrr    t0, mepc", "    li      t1, MB_TRAP_EPC", "    sw      t0, 0(t1)",
            "    csrr    t0, mtval", "    li      t1, MB_TRAP_TVAL", "    sw      t0, 0(t1)",
            "    li      t0, ST_FAIL", "    li      t1, MB_STATUS", "    sw      t0, 0(t1)",
            "    li      t0, DONE_MAGIC", "    li      t1, MB_DONE", "    sw      t0, 0(t1)",
            "8:  wfi", "    j       8b"]


def data_sections(c, tol):
    L = ["", "    # ---- data: addresses fixed by prog.ld ----",
         '    .section .data.mbox, "aw"', "    .word 0, 0, -1, 0      # done, status, fail_index, fail_count",
         f"    .dword 0x{f64_bits(tol):016X}   # tol = gamma(G), binary64, patchable",
         "    .word 0, 0, 0          # trap_cause, trap_epc, trap_tval",
         '    .section .data.a, "a"', '    .incbin "a.bin"',
         '    .section .data.b, "a"', '    .incbin "b.bin"',
         '    .section .data.c, "a"', f"    .rept {c.M * c.N}", f"    .word 0x{C_PRESET:08X}", "    .endr"]
    if c.bitexact:
        L += ['    .section .data.expect, "a"', '    .incbin "c_expect.bin"']
    else:
        L += ['    .section .data.expect, "a"', '    .incbin "c_exact.bin"',
              '    .section .data.absum, "a"', '    .incbin "c_absum.bin"']
    return L


def emit_prog(c, base, d, tol, stride_a, stride_b):
    L = header(c, base) + prologue(trap=True)
    L += gemm(c, stride_a, stride_b)
    L += [""] + cache_inval_c(c) + [""]
    L += selfcheck_bits(c.M * c.N) if c.bitexact else selfcheck_bound()
    L += store_result()
    L += uart(c) + ["", "park:", "    wfi", "    j       park"]
    L += trap_handler() + data_sections(c, tol)
    (d / "prog.S").write_text("\n".join(L) + "\n", encoding="ascii")


def emit_ld(c, base, d):
    secs = [("text", ".text", "*(.text.init) *(.text*)"), ("mbox", ".data.mbox", "*(.data.mbox)"),
            ("a", ".data.a", "*(.data.a)"), ("b", ".data.b", "*(.data.b)"), ("c", ".data.c", "*(.data.c)"),
            ("expect", ".data.expect", "*(.data.expect)")]
    if not c.bitexact:
        secs.append(("absum", ".data.absum", "*(.data.absum)"))
    txt = [f"/* Generated - case {c.cid} {c.name}. Addresses match meta.json \"segments\". */",
           "ENTRY(_start)", "SECTIONS", "{"]
    for seg, name, inp in secs:
        txt += [f"    . = 0x{base + SEG[seg]:08X};", f"    {name} : {{ {inp} }}", ""]
    txt += ["    /DISCARD/ : { *(.comment) *(.riscv.attributes) *(.note*) }", "}", ""]
    (d / "prog.ld").write_text("\n".join(txt), encoding="ascii")


# ═══════════════════════════ build one case ═══════════════════════════

def build(c, base, out):
    rng = Rng(c.seed)
    d = out / "cases" / c.dirname
    d.mkdir(parents=True, exist_ok=True)
    for stale in ("c_expect.bin", "c_exact.bin", "c_absum.bin", "a.bin"):
        (d / stale).unlink(missing_ok=True)

    sa_el, sb_el = c.K + c.pad_a, c.K + c.pad_b
    if c.data == "case008":
        A, B = gen_case008(c.M, c.K, rng), gen_case008(c.N, c.K, rng)
    elif c.kind == "exact_int":
        A, B = gen_exact_int(c, rng)
    elif c.kind == "single_round":
        A, B = gen_single_round(c, rng)
    else:
        A, B = gen_zero_mean(c.M, c.K, rng), gen_zero_mean(c.N, c.K, rng)

    cstar, absum = reference(A, B, c.M, c.N, c.K)
    meta_extra = {}
    expect = None
    if c.bitexact:
        if c.kind == "single_round":
            exact = exact_sums(A, B, c.M, c.N, c.K)
            expect, stats = single_round_stats(c, exact)
            meta_extra["rounding_coverage"] = stats
        else:
            exact = [Fraction(int(round(v))) for v in cstar]
            expect = [round_to_f32(x) for x in exact]
        for x, v, s in zip(exact, cstar, absum):
            assert Fraction(v) == x, "reference must be exact for a bit-exact case"
            if c.kind == "exact_int":
                assert s < 2 ** 24, "sum|a*b| must stay below 2^24"
        if c.kind == "exact_int":
            big_odd = sum(1 for x in exact if abs(x) >= 2 ** 23 and int(abs(x)) % 2 == 1)
            meta_extra["outputs_needing_24_bits"] = big_odd
            assert big_odd >= 1000
        if c.kind == "single_round":
            for key in ("differs_under_rtz", "differs_under_rmm", "differs_under_rup",
                        "differs_under_rdn", "differs_with_23_bit_accumulator", "exact_zero"):
                assert stats[key] >= 100, f"{key}: too few outputs to detect it"

    # files
    write_u32(d / "a.bin", bf16_words(A, sa_el))
    write_u32(d / "b.bin", bf16_words(B, sb_el))
    write_f64(d / "c_exact.bin", cstar)
    write_f64(d / "c_absum.bin", absum)
    if expect is not None:
        write_u32(d / "c_expect.bin", expect)

    tol = gamma(c.K)
    for seg, size in (("a", c.M * sa_el * 2), ("b", c.N * sb_el * 2), ("c", c.M * c.N * 4),
                      ("expect", c.M * c.N * (4 if c.bitexact else 8)),
                      ("absum", 0 if c.bitexact else c.M * c.N * 8)):
        assert size <= SEG_SIZE[seg], f"{c.cid}: {seg} does not fit"
    emit_prog(c, base, d, tol, sa_el * 2, sb_el * 2)
    emit_ld(c, base, d)

    files = {"program": ["prog.S", "prog.ld", "prog.elf", "prog.hex"],
             "A": "a.bin (BF16, row stride in stride_bytes)",
             "B": "b.bin (BF16, row stride in stride_bytes)",
             "C_star": "c_exact.bin (FP64, exact sum of the products)",
             "absum": "c_absum.bin (FP64, sum of |products|)"}
    if c.bitexact:
        files["expected"] = "c_expect.bin (FP32 bit patterns, bit-exact criterion)"

    if c.bitexact:
        criterion = {"type": "bit-exact",
                     "why": ("integer data, sum|a*b| < 2^24: no rounding in any order"
                             if c.kind == "exact_int" else
                             "two non-zero products per output: exactly one rounding in any order"),
                     "rounding_mode": "round to nearest, ties to even"}
    else:
        criterion = {"type": "error-bound",
                     "formula": "|C_hat - C*| <= gamma(G) * sum|a*b|,  gamma(G) = G*u/(1 - G*u),  u = 2^-24",
                     "default_rounds_G": c.K,
                     "default_note": "G = K holds for any FP32 accumulation; use the rounding count "
                                     "of your implementation for a tighter check (check.py --rounds/--group)",
                     "self_check_tol": tol}

    meta = {
        "id": c.cid, "name": c.name, "title": c.title, "purpose": c.purpose,
        "shape": {"M": c.M, "N": c.N, "K": c.K},
        "tile": {"mtilem": c.M, "mtilen": c.N, "mtilek": min(c.K, K_INSN), "instructions": c.n_insn},
        "csr": {k: f"0x{v:03x}" for k, v in CSR.items()},
        "layout": "C = A * B^T; A is [M][K], B is [N][K] (not [K][N]); all row-major",
        "dtype": {"A": "bf16", "B": "bf16", "C": "fp32"},
        "stride_bytes": {"A": sa_el * 2, "B": sb_el * 2, "C": c.N * 4},
        "data": {"generator": c.data, "seed": f"0x{c.seed:08X}"},
        "criterion": criterion,
        "cancellation_ratio": cancellation(cstar, absum, c.M, c.N),
        "segments": {k: f"0x{base + v:08X}" for k, v in SEG.items()
                     if not (c.bitexact and k == "absum")},
        "mailbox": {k: f"0x{base + SEG['mbox'] + v:08X}" for k, v in MBOX.items()},
        "markers": {"done": f"0x{DONE_MAGIC:08X}", "pass": f"0x{ST_PASS:08X}",
                    "fail": f"0x{ST_FAIL:08X}", "c_preset": f"0x{C_PRESET:08X}"},
        "files": files,
    }
    meta.update(meta_extra)
    (d / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"  [{c.cid}] {c.name:20s} M={c.M:<4d} N={c.N:<4d} K={c.K:<5d} "
          f"insn={c.n_insn:<3d} {criterion['type']}")
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", default="0x80000000", help="DRAM base address of the image")
    ap.add_argument("--only", help="generate a single case id")
    ap.add_argument("--out", default=str(REPO), help="repository root to write into")
    args = ap.parse_args()
    base = int(args.base, 0)
    out = Path(args.out)
    cases = [c for c in CASES if not args.only or c.cid == args.only]
    if not cases:
        raise SystemExit(f"no case {args.only}")
    print(f"generating {len(cases)} case(s), base 0x{base:08X}")
    metas = [build(c, base, out) for c in cases]
    if not args.only:
        index = {"base": f"0x{base:08X}",
                 "cases": [dict({k: m[k] for k in ("id", "name", "title", "shape")},
                                criterion=m["criterion"]["type"]) for m in metas]}
        (out / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

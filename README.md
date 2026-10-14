# riscv-matrix-gemm-vectors

[中文说明](README.zh-CN.md)

Test vectors for the BF16 × BF16 → FP32 matrix multiply-accumulate of a RISC-V matrix
extension. Each case is a bare-metal program that runs the multiplication, checks its
own result and writes a verdict to memory. The input matrices, an
implementation-independent reference and a checker for dumped results come with it.

The programs use the instructions of the
[RISC-V Matrix Extension Specification Proposal v0.6.0](https://riscv.atlassian.net/wiki/spaces/AMEX/pages/893681697/Proposals).
The reference values, the pass criterion and `check.py` do not depend on the
instruction set and can be used with other matrix extensions.

## Pass criterion

The specification leaves the order of accumulation to the implementation ("the
reduction is done as an unordered-sum"). Two compliant implementations therefore
produce different low-order bits in general, and a bit-for-bit comparison with a model
reports failures that are not errors. A tolerance in ULPs does not help either: when
positive and negative products cancel, a correct result can be thousands of ULPs away
from the exact sum.

The reference is the exact sum of the products, C\*, and the criterion is the standard
rounding-error bound

```
|Ĉ − C*|  ≤  γ(G) · Σ|a·b|,      γ(G) = G·u / (1 − G·u),      u = 2⁻²⁴
```

where G is the largest number of roundings that any product goes through. G = K holds
for every implementation that accumulates in FP32 and is the default. An
implementation that rounds less often can supply its own G for a tighter check.

The bound cannot tell one compliant rounding behaviour from another, so two cases are
built so that every compliant implementation produces the same bits. They are checked
bit for bit and detect wrong rounding and a too-narrow accumulator.
[docs/METHOD.md](docs/METHOD.md) gives the derivation, how to choose G, and what each
criterion detects and misses.

## Cases

| Case | M × N × K | Instructions | Criterion | What it tests |
|---|---|---:|---|---|
| 001 `full_tile` | 128 × 128 × 32 | 1 | error bound | every output of a full tile; run this case first |
| 002 `m1_gemv` | 1 × 128 × 1024 | 32 | error bound | a single row of A (the matrix-vector shape of token-by-token LLM decoding) |
| 003 `partial_m` | 7 × 128 × 32 | 1 | error bound | M not a power of two |
| 004 `k_multiblock` | 128 × 128 × 256 | 8 | error bound | accumulation carried across instructions |
| 005 `partial_n` | 128 × 13 × 32 | 1 | error bound | partial N |
| 006 `partial_k` | 128 × 128 × 17 | 1 | error bound | partial K: unused lanes must contribute zero |
| 007 `stride_gap` | 128 × 128 × 32 | 1 | error bound | row stride 16 elements longer than K |
| 008 `outlier_activations` | 128 × 128 × 1024 | 32 | error bound | ±6912 outliers, cancellation ratio up to 1.5 × 10⁵ |
| 009 `exact_integer` | 128 × 128 × 1024 | 32 | bit-exact | integer data, no rounding in any order; all 24 significand bits needed |
| 010 `single_rounding` | 128 × 128 × 64 | 2 | bit-exact | exactly one rounding per output: ties, sticky bits, exact cancellation |

*Instructions* is the number of `mfmacc.s.bf16` instructions that accumulate into the
same accumulator (one per 32 elements of K).

Each `cases/NNN_name/` directory contains:

| File | Contents |
|---|---|
| `prog.S`, `prog.ld` | program source and linker script (generated) |
| `prog.elf`, `prog.hex` | the assembled program. `prog.hex` is the memory image from the base address, one little-endian 32-bit word per line, for `$readmemh` |
| `a.bin`, `b.bin` | A as [M][K] and B as [N][K], BF16, row-major; row strides in `meta.json` |
| `c_exact.bin` | C\*, the exact sum of the products, FP64 [M][N] |
| `c_absum.bin` | Σ\|a·b\|, FP64 [M][N] |
| `c_expect.bin` | expected FP32 bit patterns (bit-exact cases only) |
| `meta.json` | shape, strides, addresses, criterion, cancellation ratios, data seed |

## Running a case

1. Load `prog.hex` at 0x80000000, or load `prog.elf`. The image contains the program,
   the inputs and the reference used by the self-check, and spans at most 896 KiB.
2. Start hart 0 in M-mode at 0x80000000. Other harts wait in `wfi`. No firmware and no
   device other than memory is used.
3. Wait until the word at 0x8000F000 (`done`) reads 0x600D600D.
4. Read the word at 0x8000F004 (`status`): 0x50415353 (`PASS`) or 0x4641494C (`FAIL`).

For a diagnosis, dump C (M·N·4 bytes from 0x80090000) and run `check.py` on it, or
dump the 36-byte mailbox and pass it with `--mailbox`.

An unexpected exception is caught: the program writes `mcause`, `mepc` and `mtval` to
the mailbox, sets `status` to FAIL and sets `done`. If `done` is never set, the program
hung.

To use another base address, regenerate with
`python3 generator/gen_vectors.py --base <address>` and rebuild.

### Memory map

| Offset from base | Region | Contents |
|---|---|---|
| 0x00000 | code | program text |
| 0x0F000 | mailbox | verdict, tolerance, exception record (36 bytes) |
| 0x10000 | A | BF16, up to 256 KiB |
| 0x50000 | B | BF16, up to 256 KiB |
| 0x90000 | C | FP32 [M][N], written by the program; preset to 0xA5A5A5A5 so that elements the store did not reach can be recognised |
| 0xA0000 | expect | C\* in FP64 (error-bound cases) or the expected FP32 bits (bit-exact cases) |
| 0xC0000 | absum | Σ\|a·b\| in FP64 (error-bound cases) |

### Mailbox

| Offset | Field | Meaning |
|---|---|---|
| 0x00 | `done` | 0x600D600D when the program has finished, also after an exception |
| 0x04 | `status` | 0x50415353 (`PASS`) or 0x4641494C (`FAIL`) |
| 0x08 | `fail_index` | first failing element, m·N + n; −1 if none |
| 0x0C | `fail_count` | number of failing elements |
| 0x10 | `tol` | γ(G) used by the self-check, FP64 (error-bound cases) |
| 0x18 | `trap_cause` | `mcause` of an unexpected exception; 0 if none |
| 0x1C | `trap_epc` | `mepc`, low 32 bits |
| 0x20 | `trap_tval` | `mtval`, low 32 bits |

## check.py

```
python3 check.py 001 c_dump.hex                # one 32-bit hex word per line ($writememh)
python3 check.py 001 c_dump.bin                # raw little-endian bytes
python3 check.py 004 c_dump.hex --group 16     # the implementation rounds once per 16 products
python3 check.py 004 c_dump.hex --rounds 8     # ... or give G directly
python3 check.py 001 --mailbox mbox.bin        # verdict and exception record from the mailbox
```

Python 3.8 or later, standard library only. Exit status 0 means pass, 1 fail.
`--acc-bits p` sets u = 2⁻ᵖ for an accumulator other than FP32.

For error-bound cases the script prints the largest error as a fraction of the bound,
how many elements equal the correctly rounded exact sum, and the cancellation ratios of
the case. On failure it lists the first failing elements and recognises common
patterns: C not stored, all zeros, only row 0 correct, missing rows or columns, a
transposed result (A and B swapped). For bit-exact cases it also compares the
mismatches with truncation, ties-away, directed rounding and 23- or 22-bit
accumulators, and names the one that explains them.

Example:

```
case        008 outlier_activations  (Outliers and cancellation)
shape       M=128 N=128 K=1024, 32 instruction(s)
criterion   error-bound
            |C_hat - C*| <= gamma(G) * sum|a*b|   G = 1024, u = 2^-24, gamma = 6.104e-05
worst       3.555e-03 of the bound   (these errors would still pass with G as small as 4; this is not the implementation's rounding count)
for info    4380/16384 elements equal the correctly rounded exact sum; the largest distance from it is 20627 ULP
            cancellation ratio sum|a*b| / |C*|: median 7.164, max 145720.6 - ULP counts are not a usable criterion here

>>> PASS
```

## Tightening the on-target tolerance

The programs are built with G = K. If the rounding count of the implementation is
known:

```
python3 set_tolerance.py all --group 16    # G = ceil(K/16) for every error-bound case
python3 set_tolerance.py 004 --rounds 8    # explicit G for one case
python3 set_tolerance.py all --reset       # back to G = K
```

The script rewrites the `tol` word in `prog.hex` and `prog.elf` in place; no
toolchain is needed. Bit-exact cases are left unchanged.

## ISA requirements and assumptions

* **Base ISA.** RV64GC in M-mode, bare metal. F and D are used by the self-check. The
  vector extension is not used.
* **Matrix instructions.** `mzero`, `mlae16`, `mlbe16`, `mfmacc.s.bf16` (Zmbf16) and
  `msce32`; CSRs `mtilem` (0x803), `mtilen` (0x804) and `mtilek` (0x805).
* **Tile size.** One tile covers up to 128 × 128 outputs and one instruction covers
  32 BF16 elements of K (`mtilem`, `mtilen` ≤ 128, `mtilek` ≤ 32). The specification
  requires an illegal-instruction exception for tile sizes an implementation does not
  support, and the mailbox records it. Smaller tiles need a tiled version of `gemm()`
  in the generator.
* **Operands.** A is loaded with `mlae16` into `tr0` and B with `mlbe16` into `tr1`;
  `mfmacc.s.bf16 acc0, tr1, tr0` computes acc0 += A · Bᵀ (syntax `md, ms2, ms1`). B is
  stored as [N][K], the layout `mlbe16` reads.
* **Matrix context status.** The programs set `mstatus.MS` to Initial before the first
  matrix instruction and to Clean after the last one; `mrelease` is not used. They
  assume MS at bits 30:29. The specification does not fix the position; for other bits,
  change `prologue()` and `matrix_release()` in the generator.
* **Rounding mode.** The programs do not write the matrix rounding mode (`xmfrm`,
  0x809) and assume round-to-nearest-even after reset. Otherwise add `csrw 0x809, zero`
  to `prologue()`. `frm` is set to round-to-nearest-even for the scalar self-check.

### Optional serial output

With `UART_BASE` set as well (for example `UART_BASE=0x10000000`), `build.sh` also
builds `prog_uart.elf`, which prints the verdict on a 16550-compatible UART with
byte-wide registers. For 32-bit registers on a
4-byte pitch set `UART_FLAGS="-DUART_REG_WORD -DUART_LSR_OFF=0x14"`. For boards whose
memory is loaded or read behind the caches, this build also runs `fence.i` at entry,
invalidates C with `cbo.inval` before the self-check and writes the mailbox back with
`cbo.clean`, so it needs Zicbom.

## Rebuilding

```
python3 generator/gen_vectors.py                              # cases/ and index.json
CROSS=riscv64-unknown-elf- MARCH=rv64gc_<matrix-ext> bash generator/build.sh
```

The generator needs Python 3.8 or later and no packages; it reproduces the committed
data bit for bit. `build.sh` needs a GNU toolchain whose assembler accepts the matrix
instructions above; `MARCH` is an ISA string with the toolchain's name for the matrix
extension. The committed binaries were assembled from the committed `prog.S`; rebuild
them if your toolchain encodes the matrix instructions differently.

## Self-test

`python3 tests/selftest.py` (needs numpy, about 10 s) checks that

* the generator reproduces every committed file byte for byte;
* five accumulation schemes that are all compliant (sequential, reversed, groups of 16
  or 32 exact products, pairwise tree) pass every case, with G = K and with their own
  tighter G, and produce identical bits on the bit-exact cases;
* injected faults are reported: one element off by 1% (at the output with the least
  cancellation), the last instruction missing, the accumulator not cleared, A and B
  swapped, and on the bit-exact cases truncation, ties-away rounding and a 23-bit
  accumulator.

## Not covered

* data types other than BF16 → FP32, transposed loads, element-wise and conversion
  instructions
* outputs larger than one tile
* floating-point exception flags, overflow, subnormals, NaN and infinity
* rounding modes other than round-to-nearest-even
* memory ordering and cache coherence between the matrix unit and other agents
* performance

## Repository layout

```
cases/NNN_name/     one directory per case
index.json          list of the cases
check.py            offline checker (standard library only)
set_tolerance.py    sets the tolerance of the on-target self-check
generator/          gen_vectors.py (data, reference, programs) and build.sh
tests/selftest.py   reproducibility, compliant models and fault injection
docs/METHOD.md      reference, criterion and test data in detail
```

## License

Copyright 2026 Beijing ESWIN Computing Technology Co., Ltd.

Licensed under the Apache License, Version 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).

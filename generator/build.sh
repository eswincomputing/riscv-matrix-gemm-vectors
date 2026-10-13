#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Beijing ESWIN Computing Technology Co., Ltd.
#
# Assemble every case into prog.elf and prog.hex (and optionally prog_uart.elf).
#
#   CROSS=riscv64-unknown-elf- MARCH=<isa string> bash generator/build.sh   # all cases
#   CROSS=... MARCH=... bash generator/build.sh 010                          # one case
#   CROSS=... MARCH=... UART_BASE=0x10000000 bash generator/build.sh         # + prog_uart.elf
#
# MARCH must enable RV64GC and the matrix extension, using the name your toolchain
# gives the matrix extension (for example rv64gc_<matrix-ext>). The UART build also
# uses cbo.clean and cbo.inval, so add Zicbom (rv64gc_zicbom_<matrix-ext>).
# UART_FLAGS may add -DUART_REG_WORD -DUART_LSR_OFF=0x14 for 32-bit UART registers.
#
# prog.hex is the complete memory image from the base address, one little-endian
# 32-bit word per line, readable with $readmemh. prog.bin can be produced with
#   ${CROSS}objcopy -O binary --gap-fill 0 prog.elf prog.bin
#
# The ELF keeps the program's own labels for debugging. ISA mapping symbols and the
# assembler's temporary file name are removed, so the ELF depends only on the sources
# and the toolchain version.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${CROSS:?set CROSS to the toolchain prefix, e.g. riscv64-unknown-elf-}"
: "${MARCH:?set MARCH to an ISA string with RV64GC and the matrix extension}"
ONLY="${1:-}"
CFLAGS=(-march="$MARCH" -mabi=lp64d -mcmodel=medany -ffreestanding -nostdlib -nostartfiles
        -Wl,--build-id=none -Wl,--no-warn-rwx-segments)
STRIP=(--wildcard --strip-symbol='$x*' --strip-symbol='$d*' --strip-symbol='*.o')
ok=0; fail=0

for d in "$ROOT"/cases/*/; do
    name="$(basename "$d")"
    [ -z "$ONLY" ] || [[ "$name" == "$ONLY"* ]] || continue
    # Build inside the case directory so .incbin finds the data files.
    if ! ( cd "$d" && "${CROSS}gcc" "${CFLAGS[@]}" -T prog.ld prog.S -o prog.elf ); then
        echo "  [FAIL] $name"; fail=$((fail + 1)); continue
    fi
    "${CROSS}objcopy" "${STRIP[@]}" "$d/prog.elf"
    "${CROSS}objcopy" -O binary --gap-fill 0 "$d/prog.elf" "$d/prog.bin.tmp"
    { echo "// $name: memory image from the base address, 32-bit little-endian words"
      od -An -tx4 -v -w4 "$d/prog.bin.tmp" | tr -d ' ' | grep -v '^$'
    } > "$d/prog.hex"
    words=$(( $(stat -c%s "$d/prog.bin.tmp") / 4 ))
    rm -f "$d/prog.bin.tmp"
    if [ -n "${UART_BASE:-}" ]; then
        ( cd "$d" && "${CROSS}gcc" "${CFLAGS[@]}" -DWITH_UART -DUART_BASE="$UART_BASE" \
            ${UART_FLAGS:-} -T prog.ld prog.S -o prog_uart.elf )
        "${CROSS}objcopy" "${STRIP[@]}" "$d/prog_uart.elf"
    fi
    printf "  [ OK ] %-24s %7d words\n" "$name" "$words"
    ok=$((ok + 1))
done

echo "built $ok case(s), $fail failed"
[ "$fail" -eq 0 ]

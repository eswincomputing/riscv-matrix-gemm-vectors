/* Generated - case 009 exact_integer. Addresses match meta.json "segments". */
ENTRY(_start)
SECTIONS
{
    . = 0x80000000;
    .text : { *(.text.init) *(.text*) }

    . = 0x8000F000;
    .data.mbox : { *(.data.mbox) }

    . = 0x80010000;
    .data.a : { *(.data.a) }

    . = 0x80050000;
    .data.b : { *(.data.b) }

    . = 0x80090000;
    .data.c : { *(.data.c) }

    . = 0x800A0000;
    .data.expect : { *(.data.expect) }

    /DISCARD/ : { *(.comment) *(.riscv.attributes) *(.note*) }
}

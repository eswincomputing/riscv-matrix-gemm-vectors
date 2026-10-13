# riscv-matrix-gemm-vectors

[English](README.md)

RISC-V 矩阵扩展 BF16 × BF16 → FP32 矩阵乘累加的测试向量。每个用例是一个裸机程序：执行矩阵乘、
自行检查结果，并把结论写入内存。随用例提供输入矩阵、与实现无关的参考值，以及检查结果转储的脚本。

程序使用
[RISC-V Matrix Extension Specification Proposal v0.6.0](https://riscv.atlassian.net/wiki/spaces/AMEX/pages/893681697/Proposals)
中的指令。参考值、判据和 `check.py` 不依赖指令集，也可用于其他矩阵扩展。

## 判据

规范把累加顺序留给实现（"the reduction is done as an unordered-sum"）。因此两个合规实现的低位
一般不同，与某个模型逐位比对会把正确结果报成错误。按 ULP 设容差也不行：正负乘积相互抵消时，
正确结果可以离精确和几千个 ULP。

参考值取乘积的精确和 C\*，判据是标准的舍入误差界：

```
|Ĉ − C*|  ≤  γ(G) · Σ|a·b|,      γ(G) = G·u / (1 − G·u),      u = 2⁻²⁴
```

G 是任一乘积在累加过程中经历的最多舍入次数。凡是以 FP32 累加的实现都满足 G = K，这也是默认值；
舍入次数更少的实现可以给出自己的 G，做更严的检查。

误差界无法区分不同的合规舍入方式，所以另有两个用例的数据经过构造，使所有合规实现都得到相同的
位。这两个用例逐位比对，能查出舍入方式错误和累加器位宽不足。推导、G 的取法、各判据能查出和
查不出什么，见 [docs/METHOD.md](docs/METHOD.md)（英文）。

## 用例

| 用例 | M × N × K | 指令数 | 判据 | 测试内容 |
|---|---|---:|---|---|
| 001 `full_tile` | 128 × 128 × 32 | 1 | 误差界 | 整块 tile 的全部输出；请先跑这个用例 |
| 002 `m1_gemv` | 1 × 128 × 1024 | 32 | 误差界 | A 只有一行（大模型逐 token 解码时的矩阵-向量形状） |
| 003 `partial_m` | 7 × 128 × 32 | 1 | 误差界 | M 不是 2 的幂 |
| 004 `k_multiblock` | 128 × 128 × 256 | 8 | 误差界 | 跨指令累加 |
| 005 `partial_n` | 128 × 13 × 32 | 1 | 误差界 | N 不满一块 |
| 006 `partial_k` | 128 × 128 × 17 | 1 | 误差界 | K 不满一块：未用的通道必须贡献 0 |
| 007 `stride_gap` | 128 × 128 × 32 | 1 | 误差界 | 行跨距比 K 多 16 个元素 |
| 008 `outlier_activations` | 128 × 128 × 1024 | 32 | 误差界 | ±6912 的离群值，抵消比最高 1.5 × 10⁵ |
| 009 `exact_integer` | 128 × 128 × 1024 | 32 | 逐位 | 整数数据，任何顺序都不发生舍入；需要全部 24 位有效位 |
| 010 `single_rounding` | 128 × 128 × 64 | 2 | 逐位 | 每个输出恰好一次舍入：平局、粘滞位、精确抵消 |

“指令数”是累加到同一个累加器的 `mfmacc.s.bf16` 条数（K 每 32 个元素一条）。

每个 `cases/NNN_name/` 目录包含：

| 文件 | 内容 |
|---|---|
| `prog.S`、`prog.ld` | 程序源码和链接脚本（生成） |
| `prog.elf`、`prog.hex` | 汇编后的程序。`prog.hex` 是从基地址开始的内存镜像，每行一个小端 32 位字，可用 `$readmemh` 读入 |
| `a.bin`、`b.bin` | A 为 [M][K]，B 为 [N][K]，BF16，行主序；行跨距见 `meta.json` |
| `c_exact.bin` | C\*，乘积的精确和，FP64 [M][N] |
| `c_absum.bin` | Σ\|a·b\|，FP64 [M][N] |
| `c_expect.bin` | 期望的 FP32 位模式（仅逐位用例） |
| `meta.json` | 形状、跨距、地址、判据、抵消比、数据种子 |

## 运行

1. 把 `prog.hex` 加载到 0x80000000，或加载 `prog.elf`。镜像包含程序、输入和自检用的参考值，
   最大 896 KiB。
2. 让 hart 0 在 M 模式从 0x80000000 开始执行，其他 hart 停在 `wfi`。不需要固件，除内存外不用任何
   外设。
3. 等待 0x8000F000（`done`）变为 0x600D600D。
4. 读 0x8000F004（`status`）：0x50415353 为 `PASS`，0x4641494C 为 `FAIL`。

需要定位问题时，转储 C（从 0x80090000 起 M·N·4 字节）交给 `check.py`，或转储 36 字节的信箱，
用 `--mailbox` 传入。

程序会捕获意外异常：把 `mcause`、`mepc`、`mtval` 写入信箱，`status` 置为 FAIL，并置 `done`。
如果 `done` 始终没有置位，说明程序卡住了。

换基地址时，用 `python3 generator/gen_vectors.py --base <地址>` 重新生成，再重新构建。

### 内存布局

| 相对基地址 | 区域 | 内容 |
|---|---|---|
| 0x00000 | 代码 | 程序 |
| 0x0F000 | 信箱 | 结论、容差、异常记录（36 字节） |
| 0x10000 | A | BF16，最大 256 KiB |
| 0x50000 | B | BF16，最大 256 KiB |
| 0x90000 | C | FP32 [M][N]，由程序写入；预置为 0xA5A5A5A5，没写到的元素可以认出来 |
| 0xA0000 | expect | C\*（FP64，误差界用例）或期望的 FP32 位（逐位用例） |
| 0xC0000 | absum | Σ\|a·b\|（FP64，误差界用例） |

### 信箱

| 偏移 | 字段 | 含义 |
|---|---|---|
| 0x00 | `done` | 程序结束时为 0x600D600D，发生异常后也会置位 |
| 0x04 | `status` | 0x50415353（`PASS`）或 0x4641494C（`FAIL`） |
| 0x08 | `fail_index` | 第一个出错元素的序号 m·N + n；无错为 −1 |
| 0x0C | `fail_count` | 出错元素个数 |
| 0x10 | `tol` | 自检使用的 γ(G)，FP64（误差界用例） |
| 0x18 | `trap_cause` | 意外异常的 `mcause`；无异常为 0 |
| 0x1C | `trap_epc` | `mepc` 低 32 位 |
| 0x20 | `trap_tval` | `mtval` 低 32 位 |

## check.py

```
python3 check.py 001 c_dump.hex                # 每行一个 32 位十六进制字（$writememh）
python3 check.py 001 c_dump.bin                # 小端原始字节
python3 check.py 004 c_dump.hex --group 16     # 实现每 16 个乘积舍入一次
python3 check.py 004 c_dump.hex --rounds 8     # 或直接给出 G
python3 check.py 001 --mailbox mbox.bin        # 从信箱读结论和异常记录
```

需要 Python 3.8 及以上，只用标准库。退出码 0 表示通过，1 表示失败。累加器不是 FP32 时，用
`--acc-bits p` 设 u = 2⁻ᵖ。

对误差界用例，脚本输出最大误差占误差界的比例、有多少元素等于精确和的正确舍入值，以及该用例的
抵消比。失败时列出前几个出错元素，并识别常见模式：C 没有写回、全零、只有第 0 行正确、缺行或
缺列、结果是转置（A、B 操作数接反）。对逐位用例，还会把不一致的元素与截断、远离零的平局舍入、
定向舍入、23 位或 22 位累加器的结果比较，指出与哪一种吻合。

## 收紧片上自检的容差

程序按 G = K 构建。已知实现的舍入次数时：

```
python3 set_tolerance.py all --group 16    # 所有误差界用例取 G = ceil(K/16)
python3 set_tolerance.py 004 --rounds 8    # 给单个用例指定 G
python3 set_tolerance.py all --reset       # 恢复 G = K
```

脚本直接改写 `prog.hex` 和 `prog.elf` 中的 `tol`，不需要工具链。逐位用例不受影响。

## 指令集要求与假设

* **基础指令集。** RV64GC，M 模式，裸机。自检用到 F 和 D。不使用向量扩展。
* **矩阵指令。** `mzero`、`mlae16`、`mlbe16`、`mfmacc.s.bf16`（Zmbf16）、`msce32`；CSR
  `mtilem`（0x803）、`mtilen`（0x804）、`mtilek`（0x805）。
* **Tile 尺寸。** 一个 tile 最多 128 × 128 个输出，一条指令覆盖 K 方向 32 个 BF16 元素
  （`mtilem`、`mtilen` ≤ 128，`mtilek` ≤ 32）。规范要求实现对不支持的 tile 尺寸报非法指令
  异常，信箱会记录下来。tile 更小的实现需要在生成器里把 `gemm()` 改成分块版本。
* **操作数。** A 用 `mlae16` 载入 `tr0`，B 用 `mlbe16` 载入 `tr1`；
  `mfmacc.s.bf16 acc0, tr1, tr0` 计算 acc0 += A · Bᵀ（语法为 `md, ms2, ms1`）。B 按 [N][K] 存放，
  即 `mlbe16` 读取的布局。
* **矩阵上下文状态。** 程序在第一条矩阵指令前把 `mstatus.MS` 置为 Initial，最后一条之后置为
  Clean，不使用 `mrelease`。程序假定 MS 位于第 30:29 位；规范没有规定位置，位置不同时修改生成器
  中的 `prologue()` 和 `matrix_release()`。
* **舍入模式。** 程序不写矩阵舍入模式（`xmfrm`，0x809），假定复位后为就近舍入、平局取偶。
  否则在 `prologue()` 中加入 `csrw 0x809, zero`。标量自检前 `frm` 设为就近取偶。

### 可选的串口输出

运行 `build.sh` 时再设置 `UART_BASE`（例如 `UART_BASE=0x10000000`），会额外生成 `prog_uart.elf`，
在字节寄存器的 16550 兼容串口上打印结论。寄存器为 32 位、间距 4 字节时，设 `UART_FLAGS="-DUART_REG_WORD -DUART_LSR_OFF=0x14"`。
为适应绕过缓存加载或读取内存的板卡，这个版本还会在入口执行 `fence.i`，自检前用 `cbo.inval`
作废 C 的缓存行，并用 `cbo.clean` 写回信箱，因此需要 Zicbom。

## 重新生成与构建

```
python3 generator/gen_vectors.py                              # 生成 cases/ 和 index.json
CROSS=riscv64-unknown-elf- MARCH=rv64gc_<matrix-ext> bash generator/build.sh
```

生成器需要 Python 3.8 及以上，不依赖第三方包，生成的数据与仓库中的文件逐位一致。`build.sh`
需要汇编器支持上述矩阵指令的 GNU 工具链；`MARCH` 是带有该工具链矩阵扩展名称的 ISA 字符串。
仓库中的二进制由仓库中的 `prog.S` 汇编得到；如果你的工具链对矩阵指令的编码不同，请重新构建。

## 自测

`python3 tests/selftest.py`（需要 numpy，约 10 秒）检查：

* 生成器重新生成的每个文件与仓库中的逐字节一致；
* 五种合规的累加方式（顺序、逆序、每 16 或 32 个乘积精确求和、两两求和树）在 G = K 和各自更严
  的 G 下都通过全部用例，并在逐位用例上得到相同的位；
* 注入的错误都能被查出：一个元素偏差 1%（放在抵消最少的输出上）、少执行最后一条指令、累加器
  未清零、A 与 B 接反，以及逐位用例上的截断、远离零的平局舍入和 23 位累加器。

## 不覆盖的内容

* BF16 → FP32 以外的数据类型、转置载入、逐元素指令和转换指令
* 超过一个 tile 的输出
* 浮点异常标志、上溢、次正规数、NaN 和无穷
* 就近取偶以外的舍入模式
* 矩阵单元与其他部件之间的访存顺序和缓存一致性
* 性能

## 目录结构

```
cases/NNN_name/     每个用例一个目录
index.json          用例列表
check.py            离线检查脚本（只用标准库）
set_tolerance.py    设置片上自检的容差
generator/          gen_vectors.py（数据、参考值、程序）和 build.sh
tests/selftest.py   可复现性、合规模型与错误注入
docs/METHOD.md      参考值、判据和测试数据的详细说明（英文）
```

## 许可证

Apache License 2.0，见 [LICENSE](LICENSE)。

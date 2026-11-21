# Fedora with mixed GPUs

The experimental Flash-Next path keeps an sm_120 main GPU and permits up to four sm_86 expert peers.
The build option and routing changes are implemented; a CUDA build, numerical parity, five-device execution
and performance measurements on Fedora remain required. CPU policy tests cannot establish GPU support.

## Model and format boundary

| Architecture | Older expert peers | Constraint |
| --- | --- | --- |
| `qwen4exp` | sm_86 with `PEER_AMPERE`; sm_89 with `PEER_ADA` | Up to four peers; compatible integer-quantized experts |
| `deepseek41` | Rejected | Helpers require sm_120 |
| `qwen35` | Rejected | Prompt/layer helpers require sm_120 |
| `hy4flash` | Rejected | No older-peer capability advertised |

NVFP4 experts are rejected with older peers. Dense NVFP4 weights may remain on the main card.
Prompt layer pipelines, GDN splits, head splits and explicit layer splits cannot run on older peers.
`--peer-exclusive` accepts only one peer. More VRAM does not imply higher speed or a single pooled allocation.

The implementation boundary is in `CMakeLists.txt` (portable CUDA units),
`src/program/gpu_select.cpp` (device capabilities), the model adapter's `older_expert_cards()` and
`src/models/qwen4exp/load/qfn_run.cpp` (format and execution restrictions).
`kDmMaxX` counts peers after the first; three extra rings permit four expert peers.

## Inventory and toolchain

Run from the Fedora server, before changing its software. Save output outside the checkout:

```sh
cat /etc/os-release
uname -a
lscpu
numactl --hardware
free -h
lsblk -o NAME,MODEL,SIZE,ROTA,MOUNTPOINTS
nvidia-smi -L
nvidia-smi topo -m
nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,memory.total,power.limit,driver_version --format=csv
nvidia-smi -q -d PCI
nvcc --version
gcc --version
cmake --version
git rev-parse HEAD
git diff --stat
```

Use a driver supporting both GPU families and check the NVIDIA installation guide for the exact toolkit's
Fedora and host-compiler requirements. The repository CI pins CUDA 13.0.2; its
[archived installation guide](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-installation-guide-linux/index.html)
lists Fedora 42 and explains using a compatibility GCC package. This is a reproducibility reference,
not a recommendation to downgrade the server. For another toolkit, use its matching guide.
Do not bypass the compiler check with `--allow-unsupported-compiler`.

## Build

From the repository root, with the selected CUDA toolkit and supported host compiler available:

```sh
cmake -S . -B build-ampere -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DWERROR=ON \
  -DBASALT_MODELS=qwen4exp -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DPEER_AMPERE=ON -DPEER_ADA=OFF
cmake --build build-ampere --parallel 32
ctest --test-dir build-ampere -L cpu --output-on-failure
```

If the host compiler needs selecting, pass `CMAKE_C_COMPILER`, `CMAKE_CXX_COMPILER` and
`CMAKE_CUDA_HOST_COMPILER` explicitly at the first configure, using installed compatible paths.
Keep the main CUDA architecture at 120; the peer option adds sm_86 only to the portable units.
The binaries are placed in `bin/`; test executables remain in the build directory.
Preserve a baseline checkout and its binaries separately, since building another configuration in the same
checkout can overwrite `bin/`. Record both the source SHA and local patch; the SHA alone omits uncommitted changes.

## Correctness gates

Identify GPU UUIDs from the inventory. The following names are placeholders, not literal UUIDs:

```sh
CUDA_VISIBLE_DEVICES=GPU-MAIN-UUID build-ampere/dec_moe_parity --kq
CUDA_VISIBLE_DEVICES=GPU-MAIN-UUID build-ampere/dec_moe_parity --nvfp4
CUDA_VISIBLE_DEVICES=GPU-MAIN-UUID,GPU-PEER-UUID build-ampere/dec_moe_parity --kq
```

Repeat the paired test with each physical peer. Device 0 must be the main card; the test runs its peer
window kernel on device 1. Its additional routing rings are logical tiers, not proof that four physical
peers execute together. The `--nvfp4` test belongs on the main card alone.
Run the paired integer test under `compute-sanitizer --tool memcheck`, followed by `--tool synccheck`.
Capture compiler timing (`nvcc --time`) for changed CUDA units and maximum absolute/relative errors
against both the previous kernel and an independent CPU reference. The routing test alone does not
satisfy that numerical gate.

Run an actual compatible pack with the ordered five-device list; confirm all four `peer GPU` lines
report nonzero expert counts and observe computation on every GPU. A model that fits entirely on
the main GPU or a peer without space may leave a card idle; inspect the startup explanation.
Exercise partial visibility, reordered devices, OOM, cancellation and shutdown. Check long prefill,
speculative decode and concurrent requests separately. Validate host-staged transfers when P2P is unavailable.

Compare the previous and new builds on the main card with `tools/greedy_identity.py`, pinning expert
placement. Use `tools/ppl_gate.py` with the model's independent BF16 reference for the mixed setup;
record PPL, mean KLD and top-1 agreement, first on 20 chunks and then on 100 before release.
See [quality gates](precision/quality.md). Coherent generated text alone is insufficient.

## Measurements

Use UUIDs in `CUDA_VISIBLE_DEVICES` to fix physical order, main first. With five devices visible,
run these selections separately: `0`, `0,1`, `0,1,2`, `0,1,2,3`, `0,1,2,3,4`.
Do not assume the operating system's GPU 0 is the main card. For example, after setting that order:

```sh
bin/basalt-bench --hardware
bin/basalt-bench --model /path/to/compatible.basalt --gpu 0,1,2,3,4 \
  -p 4096,16384,32768 -t conv,struct,prose -r 6 --seed 1234 -o json \
  -- --resident-experts --adapt-swaps 0 --peer-adapt-swaps 0
```

Redirect JSON and stderr to separate files outside the checkout. Keep the cold first repetition separate
from the five warm repetitions; preserve the benchmark's fixture order. Alternate baseline and candidate
builds in ABBA order with equal placement for regression checks. Record model hash, quantization,
context, sampling, cache state, speculation, power limits and all engine arguments.

Report TTFT, prefill/decode rates, ms per decode window, passes/s, RAM and VRAM per GPU, PCIe transfers,
power and errors. Keep measured capacity and speed separate; this configuration has no measured speed claim.
Only test longer contexts supported by the pack. After single-request correctness, compare concurrency
2 and 4, then run the finalist for at least one hour while checking NaN/Inf, memory growth and CUDA errors.
Measure NUMA placement and local/remote RAM before applying affinity. Reserve RAM for the OS and avoid
using swap as an inference tier. Keep the previous binary and config for rollback.

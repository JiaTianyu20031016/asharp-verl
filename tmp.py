import torch
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--duration", type=int, default=60)
parser.add_argument("--size", type=int, default=8192)
args = parser.parse_args()

torch.cuda.set_device(args.gpu)
device = torch.device(f"cuda:{args.gpu}")

print("GPU:", torch.cuda.get_device_name(device), flush=True)

# 使用 FP16，优先利用 Tensor Cores
a = torch.randn(args.size, args.size, device=device, dtype=torch.float16)
b = torch.randn(args.size, args.size, device=device, dtype=torch.float16)
c = torch.empty_like(a)

# Warmup
for _ in range(5):
    torch.mm(a, b, out=c)
torch.cuda.synchronize()

start_time = time.perf_counter()
last_print = start_time

print("Starting GPU benchmark...", flush=True)

while time.perf_counter() - start_time < args.duration:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for _ in range(20):
        torch.mm(a, b, out=c)

    end.record()
    end.synchronize()  # 确保 GPU 已经完成这批计算

    gpu_seconds = start.elapsed_time(end) / 1000
    tflops = (20 * 2 * args.size**3) / gpu_seconds / 1e12

    now = time.perf_counter()
    if now - last_print >= 2:
        print(f"GPU throughput: {tflops:.2f} TFLOPS", flush=True)
        last_print = now

print("Result finite:", torch.isfinite(c).all().item())
print("Benchmark finished.")
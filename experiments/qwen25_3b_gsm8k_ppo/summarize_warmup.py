"""Check that the critic's online explained variance has roughly stabilized."""
import re
import sys
from collections import defaultdict
from pathlib import Path

by_step = {}
for name in sys.argv[1:]:
    path = Path(name)
    if not path.exists():
        continue
    for line in path.open(errors="replace"):
        step = re.search(r"training/global_step:(\d+)", line)
        ev = re.search(r"critic/vf_explained_var:([-+0-9.eE]+)", line)
        loss = re.search(r"critic/vf_loss:np\.float64\(([-+0-9.eE]+)\)", line)
        if step and ev and loss:
            by_step[int(step.group(1))] = (float(ev.group(1)), float(loss.group(1)))

if not by_step:
    print("No critic metrics found", file=sys.stderr)
    sys.exit(1)

by_epoch = defaultdict(list)
for step, pair in sorted(by_step.items()):
    by_epoch[(step - 1) // 77].append(pair)
for epoch, values in sorted(by_epoch.items()):
    mean_ev = sum(x[0] for x in values) / len(values)
    mean_loss = sum(x[1] for x in values) / len(values)
    print(f"epoch={epoch + 1} steps={len(values)} ev={mean_ev:.4f} vf_loss={mean_loss:.4f}")

latest = max(by_step)
if latest < 616:
    print(f"Warmup incomplete: step {latest}", file=sys.stderr)
    sys.exit(1)
last = [by_step[s][0] for s in range(latest - 76, latest + 1) if s in by_step]
prev = [by_step[s][0] for s in range(latest - 153, latest - 76) if s in by_step]
if len(last) < 70 or len(prev) < 70:
    print("Insufficient full epochs for convergence check", file=sys.stderr)
    sys.exit(1)
mean_last = sum(last) / len(last)
mean_prev = sum(prev) / len(prev)
print(f"latest_step={latest} final_ev={mean_last:.4f} preceding_ev={mean_prev:.4f}")
if mean_last < 0.25 or mean_last - mean_prev > 0.08:
    print("Critic has not approximately converged", file=sys.stderr)
    sys.exit(1)

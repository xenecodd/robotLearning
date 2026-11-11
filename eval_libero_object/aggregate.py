"""Aggregate per-task eval_info_task*.json of step B into results.json + results.md (Wilson 95% CI)."""
import glob
import json
import math
import re

EXP = "experiments/2026-10-12_smolvla_libero_object_suite_10x10"


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


rows = []
for f in sorted(glob.glob(f"{EXP}/eval_info_task*.json"), key=lambda s: int(re.search(r"task(\d+)", s).group(1))):
    i = int(re.search(r"task(\d+)", f).group(1))
    d = json.load(open(f))
    o = d["overall"]
    n = int(o["n_episodes"])
    k = round(o["pc_success"] * n / 100)
    task = next(iter(d.get("per_task", [{}])), {}).get("task", "")
    rows.append({"task_id": i, "task": task, "success": k, "n": n})

K = sum(r["success"] for r in rows)
N = sum(r["n"] for r in rows)
lo, hi = wilson(K, N)
json.dump({"per_task": rows, "overall": {"success": K, "n": N, "wilson95": [lo, hi]}}, open(f"{EXP}/results.json", "w"), indent=2)

lines = ["| task | success | % | Wilson 95% CI |", "|---|---|---|---|"]
for r in rows:
    a, b = wilson(r["success"], r["n"])
    lines.append(f"| {r['task_id']} | {r['success']}/{r['n']} | {100 * r['success'] / r['n']:.0f} | {100 * a:.0f}–{100 * b:.0f} |")
lines.append(f"| **total** | **{K}/{N}** | **{100 * K / N:.1f}** | **{100 * lo:.1f}–{100 * hi:.1f}** |")
open(f"{EXP}/results.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

import json
from pathlib import Path
import numpy as np


def summarize_results(results, crash_count, restart_count, recovery_times):
    total = len(results)
    ok = sum(1 for r in results if r and r.get("status") == "ok")
    failed = total - ok

    latencies = [
        r["latency_sec"]
        for r in results
        if r and r.get("status") == "ok"
    ]

    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    p95_latency = float(np.percentile(latencies, 95)) if latencies else 0.0
    avg_recovery_time = float(np.mean(recovery_times)) if recovery_times else 0.0

    return {
        "total_requests": total,
        "successful_requests": ok,
        "failed_requests": failed,
        "success_rate": ok / max(total, 1),
        "avg_latency_sec": avg_latency,
        "p95_latency_sec": p95_latency,
        "worker_crashes_detected": crash_count,
        "worker_restarts": restart_count,
        "avg_recovery_time_sec": avg_recovery_time,
        "recovery_times_sec": recovery_times,
    }


def print_metrics(metrics):
    print("\n" + "=" * 80)
    print("LLM Serving Reliability Demo Metrics")
    print("=" * 80)
    print(f"Total requests:          {metrics['total_requests']}")
    print(f"Successful requests:     {metrics['successful_requests']}")
    print(f"Failed requests:         {metrics['failed_requests']}")
    print(f"Success rate:            {metrics['success_rate'] * 100:.2f}%")
    print(f"Average latency:         {metrics['avg_latency_sec']:.4f}s")
    print(f"P95 latency:             {metrics['p95_latency_sec']:.4f}s")
    print(f"Crashes detected:        {metrics['worker_crashes_detected']}")
    print(f"Worker restarts:         {metrics['worker_restarts']}")
    print(f"Avg recovery time:       {metrics['avg_recovery_time_sec']:.4f}s")
    print("=" * 80 + "\n")


def save_metrics(metrics, path="metrics/results.json"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))
    print(f"[metrics] saved to {path}")

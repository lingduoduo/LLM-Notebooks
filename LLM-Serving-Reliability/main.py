import argparse
import time

from scheduler import ReliabilityScheduler
from metrics import summarize_results, print_metrics, save_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--failure_rate", type=float, default=0.08)
    parser.add_argument("--timeout_sec", type=float, default=2.0)
    parser.add_argument("--sleep_between_requests", type=float, default=0.01)
    return parser.parse_args()


def main():
    args = parse_args()

    scheduler = ReliabilityScheduler(
        num_workers=args.workers,
        failure_rate=args.failure_rate,
    )
    scheduler.start()

    results = []

    try:
        print(
            f"[demo] starting: workers={args.workers}, "
            f"requests={args.requests}, failure_rate={args.failure_rate}"
        )

        for request_id in range(args.requests):
            scheduler.monitor_once()

            prompt = f"Explain distributed LLM serving reliability. request={request_id}"

            try:
                scheduler.route_request(request_id, prompt)
            except Exception as e:
                print(f"[scheduler] failed to route request={request_id}: {e}")
                results.append({
                    "request_id": request_id,
                    "status": "routing_failed",
                    "latency_sec": 0.0,
                })
                continue

            result = scheduler.collect_result(timeout_sec=args.timeout_sec)

            if result is None:
                print(f"[client] request={request_id} timed out")
                results.append({
                    "request_id": request_id,
                    "status": "timeout",
                    "latency_sec": args.timeout_sec,
                })
            else:
                if result["status"] != "ok":
                    print(
                        f"[client] request={request_id} failed with "
                        f"status={result['status']}, worker={result.get('worker_id')}"
                    )
                results.append(result)

            scheduler.monitor_once()
            time.sleep(args.sleep_between_requests)

    finally:
        metrics = summarize_results(
            results=results,
            crash_count=scheduler.crash_count,
            restart_count=scheduler.restart_count,
            recovery_times=scheduler.recovery_times,
        )
        print_metrics(metrics)
        save_metrics(metrics)
        scheduler.stop()


if __name__ == "__main__":
    main()

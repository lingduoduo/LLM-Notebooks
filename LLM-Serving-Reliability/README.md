# LLM Serving Reliability Demo

A clean, runnable demo for **cluster scheduling & stability** concepts:

- Worker isolation
- Request routing
- Fault injection
- Auto-recovery / self-healing
- Latency, success-rate, and recovery-time metrics

This is designed to be runnable on a laptop or Google Colab without Kubernetes, RDMA, or NCCL hardware.

## What This Simulates

| Real Infra Concept | Demo Equivalent |
|---|---|
| Cluster scheduler | Python scheduler + request queue |
| Pod / process isolation | Independent worker processes |
| Fault self-healing | Monitor detects dead workers and restarts them |
| Load balancing | Round-robin routing to healthy workers |
| Service availability | Success rate under injected failures |
| Recovery SLO | Time from crash detection to worker restart |

## Install

```bash
pip install -r requirements.txt
```

## Run Basic Demo

```bash
python main.py --workers 3 --requests 100 --failure_rate 0.08
```

## Run Without Failures

```bash
python main.py --workers 3 --requests 100 --failure_rate 0.0
```

## Run Stress Test

```bash
python main.py --workers 5 --requests 500 --failure_rate 0.12
```

## Outputs

The script prints:

- total requests
- successful requests
- failed requests
- success rate
- average latency
- p95 latency
- worker crash count
- worker restart count
- average recovery time

It also saves results to:

```text
metrics/results.json
```

## Resume Bullet

Built a fault-tolerant LLM serving reliability simulator with worker isolation, load-balanced request routing, fault injection, and auto-recovery, improving service availability under simulated worker failures while tracking success rate, latency, and recovery-time SLOs.

#!/usr/bin/env python3
"""Intelligent Gateway Test"""
from typing import Any

import requests

BASE_URL = "http://localhost:8000"
HEALTH_TIMEOUT = 5
REQUEST_TIMEOUT = 30
STATS_TIMEOUT = 10

TEST_CASES = [
    {
        "name": "Auto fast",
        "question": "What is AI?",
        "requested_model": "auto",
        "expected_model": "gpt-3.5-turbo",
    },
    {
        "name": "Auto balanced",
        "question": "Please explain the basic principles of machine learning",
        "requested_model": "auto",
        "expected_model": "gpt-4",
    },
    {
        "name": "Auto premium",
        "question": "Design a complete distributed recommendation system architecture",
        "requested_model": "auto",
        "expected_model": "o3-mini",
    },
    {
        "name": "Explicit engine type",
        "question": "What is AI?",
        "requested_model": "premium",
        "expected_model": "o3-mini",
    },
    {
        "name": "Explicit model name",
        "question": "What is AI?",
        "requested_model": "gpt-4",
        "expected_model": "gpt-4",
    },
]


def is_service_healthy(session: requests.Session) -> bool:
    """Check whether the gateway is reachable."""
    try:
        response = session.get(f"{BASE_URL}/health", timeout=HEALTH_TIMEOUT)
        return response.status_code == 200
    except requests.RequestException:
        return False


def request_completion(
    session: requests.Session, question: str, requested_model: str
) -> requests.Response:
    """Send one completion request to the gateway."""
    return session.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": requested_model,
            "messages": [{"role": "user", "content": question}],
        },
        timeout=REQUEST_TIMEOUT,
    )


def print_case_result(case: dict[str, str], result: dict[str, Any]) -> bool:
    """Show the actual routing result and whether it matched expectations."""
    actual_model = result.get("model")
    passed = actual_model == case["expected_model"]
    status = "PASS" if passed else "FAIL"

    print(f"Requested: {case['requested_model']}")
    print(f"Expected model: {case['expected_model']}")
    print(f"Actual model: {actual_model} [{status}]")

    choices = result.get("choices", [])
    if choices:
        content = choices[0].get("message", {}).get("content", "")
        print(f"Response length: {len(content)} characters")

    return passed


def print_stats(session: requests.Session) -> None:
    """Print gateway stats at the end of the run."""
    try:
        response = session.get(f"{BASE_URL}/stats", timeout=STATS_TIMEOUT)
        if response.status_code == 200:
            stats = response.json()
            print("\nSystem stats:")
            print(f"Available engines: {stats['engine_types']}")
            print(f"API status: {'Available' if stats['api_available'] else 'Simulated'}")
            return
        print(f"\nStats request failed: {response.status_code}")
    except requests.exceptions.Timeout:
        print("\nStats request timed out")
    except requests.RequestException as exc:
        print(f"\nUnable to fetch stats: {exc}")


def test_gateway():
    """Test intelligent model selection and explicit model routing."""
    passed = 0

    with requests.Session() as session:
        if not is_service_healthy(session):
            print("Gateway service is unhealthy or unreachable")
            return

        print("Gateway service is healthy")

        for index, case in enumerate(TEST_CASES, 1):
            print(f"\nTest {index}: {case['name']}")
            print(f"Question: {case['question']}")

            try:
                print("Processing request...")
                response = request_completion(
                    session, case["question"], case["requested_model"]
                )

                if response.status_code == 200:
                    result = response.json()
                    if print_case_result(case, result):
                        passed += 1
                else:
                    print(f"Request failed: {response.status_code}")
                    print(f"Response body: {response.text}")
            except requests.exceptions.Timeout:
                print("Request timed out - server response took too long")
            except requests.exceptions.ConnectionError:
                print("Connection error - unable to reach the server")
            except requests.RequestException as exc:
                print(f"Request error: {exc}")

        print(f"\nSummary: {passed}/{len(TEST_CASES)} tests passed")
        print_stats(session)

    print("\nTest completed")


if __name__ == "__main__":
    test_gateway()

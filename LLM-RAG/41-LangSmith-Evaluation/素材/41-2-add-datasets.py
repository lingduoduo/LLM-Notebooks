import os
from langsmith import Client
from datetime import datetime
import dotenv

# --- LangSmith Configuration ---
dotenv.load_dotenv()

# Initialize LangSmith client
client = Client()

# Define test cases (inputs/outputs schema should match your real application)
test_cases = [
    {
        "inputs": {"question": "How do I return or exchange an item?"},
        "outputs": {"answer": "You can request a return/exchange on the order page. Returns are accepted within 7 days (no-questions-asked)."},
        "metadata": {
            "category": "After-sales service",
            "difficulty": "Easy",
            "expected_keywords": ["order page", "7 days", "return"],
            "source": "manual",
        },
    },
    {
        "inputs": {"question": "What should I do if I can't find my order status?"},
        "outputs": {"answer": "Please provide your order number and I can help you check the current status."},
        "metadata": {
            "category": "Order tracking",
            "difficulty": "Medium",
            "expected_keywords": ["order number", "check", "status"],
            "source": "manual",
        },
    },
    {
        "inputs": {"question": "What payment methods do you support?"},
        "outputs": {"answer": "We support WeChat Pay, Alipay, bank cards, and other payment methods."},
        "metadata": {
            "category": "Payments",
            "difficulty": "Easy",
            "expected_keywords": ["WeChat", "Alipay", "bank card"],
            "source": "manual",
        },
    },
]

# Generate a unique dataset name dynamically (to avoid name collisions)
dataset_name = f"customer_service_qa_{datetime.now().strftime('%Y%m%d_%H%M')}"

try:
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Customer support Q&A test set - includes basic scenarios and production feedback samples",
        metadata={
            "version": "1.0",
            "created_by": "automation-pipeline",
            "test_scenarios": ["After-sales service", "Order tracking", "Payments"],
            "total_cases": len(test_cases),
            "collection_method": "manual_and_production",
        },
    )
    print(f"Dataset created successfully: {dataset.id} ({dataset_name})")
except Exception as e:
    if "already exists" in str(e):
        print(f"Dataset name already exists: {dataset_name}")
        print("Tip: add a timestamp or version suffix to ensure uniqueness.")
        raise
    else:
        print(f"Failed to create dataset: {e}")
        raise

# Batch add Examples
for i, case in enumerate(test_cases):
    try:
        example = client.create_example(
            dataset_id=dataset.id,
            inputs=case["inputs"],
            outputs=case["outputs"],  # reference output
            metadata=case["metadata"],
        )
        print(f"Test case {i + 1} added successfully: {example.id}")
    except Exception as e:
        print(f"Failed to add test case {i + 1}: {e}")

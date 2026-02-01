import requests
import json

def run_dify_workflow(dify_workflow_input):
    """
    Execute a Dify workflow (Chat application) and return the result.

    :param dify_workflow_input: Workflow input as a dictionary.
                                All values will be converted to strings.
                                For Chat applications, the "query" key is used
                                as the user message.
    :return: Workflow output as a dictionary, or None if an error occurs.
    """

    # Dify Chat API endpoint (local deployment)
    # url = "https://api.dify.ai/v1/workflows/run"
    url = "http://localhost/v1/chat-messages"

    headers = {
        "Authorization": "Bearer app-mMl5Qiq3Gv9yGoJGjDRjH8m6",
        "Content-Type": "application/json",
    }

    # Ensure all input values are strings
    dify_workflow_input = {k: str(v) for k, v in dify_workflow_input.items()}

    # Extract "query" for Chat apps, the rest go into "inputs"
    query = dify_workflow_input.get("query", "")
    inputs = {k: v for k, v in dify_workflow_input.items() if k != "query"}

    # Request payload
    data = {
        "inputs": inputs,
        "query": query,
        "response_mode": "blocking",
        "user": "default-user-id",
    }

    # Optional proxy configuration
    proxies = {
        "https": "http://127.0.0.1:7890",
        "http": "http://127.0.0.1:7890",
    }

    try:
        # Send request (disable proxies if not needed)
        # response = requests.post(url, headers=headers, json=data, proxies=proxies)
        response = requests.post(url, headers=headers, json=data)

        # Print full response for debugging
        print(f"Status code: {response.status_code}")
        print(f"Response text: {response.text}")

        response.raise_for_status()  # Raise an exception for non-2xx responses

        result = response.json()

        # Workflow-style response
        if "data" in result and "outputs" in result["data"]:
            return result["data"]["outputs"]

        # Chat-style response (nested)
        elif "data" in result and "answer" in result["data"]:
            return {"answer": result["data"]["answer"]}

        # Chat-style response (blocking mode, top-level)
        elif "answer" in result:
            return {"answer": result["answer"]}

        else:
            print(
                "Unexpected response structure:",
                json.dumps(result, ensure_ascii=False, indent=2),
            )
            return None

    except requests.RequestException as e:
        print(f"Request error: {str(e)}")
        return None


# Example usage
run_dify_workflow({"query": "Hello"})

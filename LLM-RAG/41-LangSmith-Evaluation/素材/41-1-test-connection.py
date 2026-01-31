import os
from langsmith import Client
import dotenv

dotenv.load_dotenv()

# Initialize LangSmith client
client = Client()

# === 1. Verify LangSmith connection ===
try:
    datasets = list(client.list_datasets(limit=1))
    print("LangSmith connection successful")
except Exception as e:
    print(f"LangSmith connection failed: {type(e).__name__}: {e}")
    exit(1)

# === 2. Verify that tracing is working ===
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke([HumanMessage(content="Hello, this is a test!")])

    print(f"Tracing is working correctly. LLM response: {response.content}")

    # Prompt the user to check the LangSmith dashboard
    project_name = os.getenv("LANGSMITH_PROJECT", "default")
    print(f"\nPlease visit https://smith.langchain.com/projects?project_name={project_name}")
    print("to view the complete trace for this invocation.")

except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please check whether langchain-openai is installed correctly.")
except Exception as e:
    print(f"LLM invocation failed: {type(e).__name__}: {e}")
    print("Please verify that OPENAI_API_KEY is valid.")

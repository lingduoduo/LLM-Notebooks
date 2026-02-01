import os
from datetime import datetime
from uuid import uuid4

from langsmith import Client
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSerializable

import dotenv


# Load environment variables from a .env file (if present)
dotenv.load_dotenv()


def create_customer_service_chain() -> RunnableSerializable:
    """
    Create a customer support response chain and return a LangChain-compatible Runnable.

    Returns:
        A callable chain whose output format is: {"answer": str}
    """
    # Prompt template
    prompt = PromptTemplate.from_template(
        """You are a professional customer support assistant. Provide an accurate and friendly response based on the user's question.

User question: {question}

Response requirements:
1. Friendly and professional tone
2. Accurate and specific information
3. If more information is needed, proactively ask follow-up questions
4. Keep it within 100 Chinese characters (or similarly concise)

Answer:"""
    )

    # LLM config (lower temperature for more consistent outputs)
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        max_tokens=200,
        timeout=10,
    )

    # Build the chained pipeline
    chain = (
        prompt
        | llm
        | RunnableLambda(lambda response: {"answer": response.content.strip()})
    )

    return chain


# Initialize the test chain
test_chain = create_customer_service_chain()

# Smoke test: ensure the chain can be invoked successfully
try:
    test_result = test_chain.invoke({"question": "How do I return or exchange an item?"})
    print(f"Unit test passed: {test_result['answer']}")
except Exception as e:
    print(f"Chain initialization failed: {type(e).__name__}: {e}")
    raise

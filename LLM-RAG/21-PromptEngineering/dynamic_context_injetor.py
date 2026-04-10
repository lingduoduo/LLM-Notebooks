# Author: Ling Huang
# Date: 2026-01-26
# v1.0 Static Context Injection
# v2.0 Dynamic Context Injection

import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Read API key (recommended)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError(
        "Missing OPENAI_API_KEY. Please set it in your environment or in a .env file."
    )

# Initialize OpenAI client (FIXED)
client = OpenAI(api_key=OPENAI_API_KEY)


def query(user_prompt: str) -> str:
    """
    Send a user prompt to the OpenAI API and return the response content.

    Args:
        user_prompt (str): User input prompt

    Returns:
        str: AI-generated response
    """
    try:
        response = client.chat.completions.create(
            model="o3-mini",
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.choices[0].message.content
    except Exception as e:
        # Keep it readable while preserving the exception
        return f"Error calling model: {e}"


class ContextTemplateFactory:
    """Context template factory responsible for creating and managing different context templates"""

    @staticmethod
    def create_customer_feedback_template():
        """Create a customer feedback analysis template"""
        return {
            "name": "Customer Feedback Analysis",
            "template": """Please extract the main issues from the customer feedback using the following categories:
1. Logistics issues: delivery delays, damaged packaging, incorrect delivery address, etc.
2. Product issues: quality defects, feature mismatch, incorrect specifications, etc.
3. Service issues: slow customer support response, poor attitude, unsatisfactory resolution, etc.
4. Pricing issues: price changes, coupon problems, refund issues, etc.

Example:
Original text: "The package hasn’t arrived after three days. Customer service said it’s on the way and was polite."
Analysis result: Delivery delay (main issue) + Fast response and good attitude from customer service (positive feedback)

Now please analyze the following customer feedback:
{user_input}""",
            "keywords": [
                "delivery",
                "customer service",
                "shipping",
                "delay",
                "attitude",
                "service",
                "logistics",
                "feedback",
                "review",
                "complaint",
                "suggestion",
            ],
        }

    @staticmethod
    def create_technical_doc_template():
        """Create a technical documentation summary template"""
        return {
            "name": "Technical Document Summary",
            "template": """Please summarize the technical document using the following structure:
1. Core concepts: extract 3–5 key technical concepts
2. Main features: list the core functionalities
3. Use cases: describe applicable business scenarios
4. Notes: highlight important limitations or considerations

Reference format:
- Core concepts: [Concept 1], [Concept 2], [Concept 3]
- Main features: [Feature description]
- Use cases: [Scenario description]
- Notes: [Important notes]

Please summarize the following content:
{user_input}""",
            "keywords": [
                "technology",
                "documentation",
                "api",
                "framework",
                "algorithm",
                "interface",
                "authentication",
                "request",
                "error handling",
            ],
        }

    @staticmethod
    def create_default_template():
        """Create a default generic template"""
        return {
            "name": "Generic Summary",
            "template": """Please summarize the following content:
{user_input}""",
            "keywords": [],
        }

    @classmethod
    def get_all_templates(cls):
        """Get all available templates"""
        return {
            "Customer Feedback Analysis": cls.create_customer_feedback_template(),
            "Technical Document Summary": cls.create_technical_doc_template(),
            "Generic Summary": cls.create_default_template(),
        }


class DynamicContextInjector:
    """
    Dynamic Context Injector – uses the Template Method pattern to define the processing workflow.

    Processing steps:
    1. Initialize template library
    2. Detect task type
    3. Select appropriate template
    4. Inject context
    5. Execute AI query
    """

    def __init__(self):
        """Initialization: load all context templates"""
        self.context_templates = self._initialize_templates()

    def _initialize_templates(self):
        """Step 1: Initialize template library"""
        print("Initializing context template library...")
        templates = ContextTemplateFactory.get_all_templates()
        print(f"Successfully loaded {len(templates)} template types")
        return templates

    def _detect_task_type(self, user_input: str) -> str:
        """Step 2: Detect task type"""
        print(f"Analyzing input content: {user_input[:30]}...")

        user_input_lower = user_input.lower()

        for template_name, template_config in self.context_templates.items():
            if template_name == "Generic Summary":
                continue

            for keyword in template_config["keywords"]:
                # keyword matching is already lowercase in technical template
                if keyword.lower() in user_input_lower:
                    print(f"Detected keyword '{keyword}', matched template: {template_name}")
                    return template_name

        print("No specific keywords found, using generic summary template")
        return "Generic Summary"

    def _select_template(self, task_type: str) -> dict:
        """Step 3: Select appropriate template"""
        print(f"Selecting template type: {task_type}")

        selected_template = self.context_templates.get(task_type)
        if not selected_template:
            print("Template not found, falling back to generic template")
            selected_template = self.context_templates["Generic Summary"]

        return selected_template

    def _inject_context(self, user_input: str, template_config: dict) -> str:
        """Step 4: Inject context"""
        print("Injecting contextual information...")

        template_content = template_config["template"]
        enhanced_prompt = template_content.format(user_input=user_input)

        print(f"Enhanced prompt length: {len(enhanced_prompt)} characters")
        return enhanced_prompt

    def _execute_query(self, enhanced_prompt: str) -> str:
        """Step 5: Execute AI query"""
        print("Calling AI model...")
        print("-" * 50)
        return query(enhanced_prompt)

    def process_with_context(self, user_input: str, task_type: str | None = None) -> str:
        """Template method: defines the complete processing workflow"""
        print("Starting dynamic context injection workflow")
        print("=" * 60)

        try:
            if task_type is None:
                detected_type = self._detect_task_type(user_input)
            else:
                detected_type = task_type
                print(f"Using specified task type: {detected_type}")

            template_config = self._select_template(detected_type)
            enhanced_prompt = self._inject_context(user_input, template_config)
            result = self._execute_query(enhanced_prompt)

            print("Workflow completed")
            return result

        except Exception as e:
            print(f"Error during processing: {str(e)}")
            return f"Processing failed: {str(e)}"

    # Backward-compatible method name
    def query_with_context(self, user_input: str, task_type: str | None = None) -> str:
        return self.process_with_context(user_input, task_type)


def demo_comparison():
    """Demonstrate comparison between basic prompt and context-enriched prompt"""
    injector = DynamicContextInjector()

    test_cases = [
        "The package hasn’t arrived after three days. Customer service said it’s on the way and was polite, but I urgently need the item.",
        "This API documentation explains how to use RESTful interfaces, including authentication, request formats, and error handling.",
    ]

    for i, test_input in enumerate(test_cases, 1):
        print(f"\n=== Test Case {i} ===")
        print(f"Original input: {test_input}")

        print("\n[Basic Prompt Result]:")
        basic_result = query(f"Summarize the following text: {test_input}")
        print(basic_result)

        print("\n[Context-Enriched Prompt Result]:")
        context_result = injector.query_with_context(test_input)
        print(context_result)

        print("\n" + "=" * 80)


if __name__ == "__main__":
    print("\n=== Dynamic Context Injection Demo ===")
    demo_comparison()

'''
(llm_clean)  🐍 llm_clean  linghuang@Mac  ~/Git/LLMs   rag-optimization  /Users/linghuang/miniconda3/envs/llm_clean/bin/python /Users/linghuang/Git/LLMs/LLM-RAG/21-Dyn
amicContext/dynamic_context_injetor.py

=== Dynamic Context Injection Demo ===
Initializing context template library...
Successfully loaded 3 template types

=== Test Case 1 ===
Original input: The package hasn’t arrived after three days. Customer service said it’s on the way and was polite, but I urgently need the item.

[Basic Prompt Result]:
The package is delayed, as it still hasn't arrived three days after shipping, and although the customer service representative was polite and confirmed it’s on its way, there is an urgent need for the item.

[Context-Enriched Prompt Result]:
Starting dynamic context injection workflow
============================================================
Analyzing input content: The package hasn’t arrived aft...
Detected keyword 'customer service', matched template: Customer Feedback Analysis
Selecting template type: Customer Feedback Analysis
Injecting contextual information...
Enhanced prompt length: 862 characters
Calling AI model...
--------------------------------------------------
Workflow completed
Analysis result:
• Logistics issue: Delivery delay ("The package hasn’t arrived after three days")
• Service feedback: Positive customer service response ("Customer service said it’s on the way and was polite")
• Note: The urgency expressed ("I urgently need the item") reinforces the impact of the delivery delay.

================================================================================

=== Test Case 2 ===
Original input: This API documentation explains how to use RESTful interfaces, including authentication, request formats, and error handling.

[Basic Prompt Result]:
The documentation details how to work with RESTful APIs, covering authentication, request formatting, and error handling.

[Context-Enriched Prompt Result]:
Starting dynamic context injection workflow
============================================================
Analyzing input content: This API documentation explain...
Detected keyword 'documentation', matched template: Technical Document Summary
Selecting template type: Technical Document Summary
Injecting contextual information...
Enhanced prompt length: 627 characters
Calling AI model...
--------------------------------------------------
Workflow completed
- Core concepts: RESTful interfaces, Authentication, Request formats, Error handling  
- Main features: Provides endpoints to authenticate users, outlines standardized request structures, and establishes error response patterns  
- Use cases: Suitable for integrating mobile, web, or backend applications that require secure, structured communication with a service through a REST API  
- Notes: Developers should follow authentication protocols carefully and ensure proper handling of error responses as specified in the documentation

================================================================================
'''
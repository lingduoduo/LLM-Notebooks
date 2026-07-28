"""
Intelligence processing tools: Code generation, reasoning, and guarding.
Based on AWorld intelligence-* servers.
"""
import json
import logging
import os
from typing import Dict, Any, List

from openai import OpenAI
from dotenv import load_dotenv

from llm_fallback import reasoning_safe_temperature, resolve_llm, token_limit_kwargs


load_dotenv()
logger = logging.getLogger(__name__)


def _parse_json_response(content) -> Dict[str, Any]:
    """Parse a JSON object out of an LLM reply, tolerating markdown fences.

    A model asked for JSON may still wrap it in a ```json ... ``` fence, which
    makes a bare json.loads() fail. Strip an optional fence and, as a last
    resort, slice from the first '{' to the last '}' before parsing.
    """
    text = (content or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


def _client_and_model():
    """Build an OpenAI-compatible client + model + route, with OpenRouter fallback.

    Uses OPENAI_API_KEY directly when present; otherwise routes through
    OPENROUTER_API_KEY. Raises RuntimeError (listing accepted keys) when neither
    is configured, so callers can surface a clear error.

    ``base_url`` is returned alongside the client because reasoning-model
    parameter renames apply only on the direct OpenAI route.
    """
    api_key, base_url, model = resolve_llm()
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    return client, model, base_url


async def generate_python_code(
    task_description: str,
    requirements: str | None = None,
    temperature: float = 0.7
) -> Dict[str, Any]:
    """
    Generate Python code based on task description.
    
    Args:
        task_description: Description of what the code should do
        requirements: Optional additional requirements
        temperature: LLM temperature for creativity
        
    Returns:
        Dictionary with generated code
    """
    try:
        try:
            client, model, base_url = _client_and_model()
        except RuntimeError as e:
            return {"success": False, "error": str(e)}
        
        prompt = f"""Generate Python code for the following task:

Task: {task_description}

{f'Requirements: {requirements}' if requirements else ''}

Provide clean, well-documented Python code that solves the task."""
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert Python programmer. Generate clean, efficient code."},
                {"role": "user", "content": prompt}
            ],
            temperature=reasoning_safe_temperature(model, temperature),
            **token_limit_kwargs(model, 2000, base_url)
        )
        
        code = response.choices[0].message.content
        
        return {
            "success": True,
            "task": task_description,
            "code": code,
            "model": model,
            "tokens_used": response.usage.total_tokens if response.usage else 0
        }
        
    except Exception as e:
        return {"success": False, "error": f"Code generation failed: {str(e)}"}


async def complex_problem_reasoning(
    problem: str,
    context: str | None = None,
    reasoning_steps: int = 3
) -> Dict[str, Any]:
    """
    Perform complex problem reasoning with step-by-step thinking.
    
    Args:
        problem: Problem statement
        context: Optional context information
        reasoning_steps: Number of reasoning steps
        
    Returns:
        Dictionary with reasoning process and conclusion
    """
    try:
        try:
            client, model, base_url = _client_and_model()
        except RuntimeError as e:
            return {"success": False, "error": str(e)}
        
        prompt = f"""Analyze the following problem with step-by-step reasoning:

Problem: {problem}

{f'Context: {context}' if context else ''}

Think through this problem step by step. Provide {reasoning_steps} clear reasoning steps, then give your conclusion."""
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert problem solver. Think step by step."},
                {"role": "user", "content": prompt}
            ],
            temperature=reasoning_safe_temperature(model, 0.7),
            **token_limit_kwargs(model, 1500, base_url)
        )
        
        reasoning = response.choices[0].message.content
        
        return {
            "success": True,
            "problem": problem,
            "reasoning": reasoning,
            "model": model,
            "tokens_used": response.usage.total_tokens if response.usage else 0
        }
        
    except Exception as e:
        return {"success": False, "error": f"Reasoning failed: {str(e)}"}


async def guard_reasoning_process(
    proposed_action: str,
    context: Dict[str, Any],
    safety_rules: List[str] | None = None
) -> Dict[str, Any]:
    """
    Guard and validate a proposed action or reasoning.
    
    Args:
        proposed_action: The action being proposed
        context: Context information for evaluation
        safety_rules: Optional list of safety rules to check
        
    Returns:
        Dictionary with safety evaluation
    """
    try:
        try:
            client, model, base_url = _client_and_model()
        except RuntimeError as e:
            return {"success": False, "error": str(e)}
        
        rules_text = "\n".join(f"- {rule}" for rule in (safety_rules or []))
        safety_rules_block = f"Safety Rules to Check:\n{rules_text}" if safety_rules else ""

        prompt = f"""Evaluate the safety and appropriateness of the following proposed action:

Proposed Action: {proposed_action}

Context: {json.dumps(context, indent=2)}

{safety_rules_block}

Analyze whether this action is:
1. Safe to execute
2. Aligned with the context and goals
3. Free from potential harmful consequences

Respond with a single JSON object and nothing else:
{{
    "approved": true or false,
    "reasoning": "Your evaluation reasoning",
    "concerns": ["Any safety concerns, empty list if none"],
    "suggestions": ["Alternative approaches if not approved"]
}}"""
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a safety validator. Carefully evaluate proposed actions."},
                {"role": "user", "content": prompt}
            ],
            temperature=reasoning_safe_temperature(model, 0.3),
            **token_limit_kwargs(model, 800, base_url)
        )
        
        evaluation = response.choices[0].message.content

        # Read the verdict from a parsed JSON field rather than by substring
        # matching. Substring matching fails open here: the prompt used to ask
        # for "approved: true/false", and that literal template text contains
        # "approved: true" but not "approved: false", so a model that merely
        # restated the requested format was scored as an approval. It also
        # failed closed on ordinary formatting variation ("**Approved:** true",
        # or a JSON body), denying verdicts that were in fact approvals.
        #
        # A guard that cannot read its own verdict must not approve, so any
        # parse failure or non-boolean value is treated as a rejection.
        try:
            parsed = _parse_json_response(evaluation)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("Guard verdict was unparseable, denying: %s", exc)
            return {
                "success": True,
                "proposed_action": proposed_action,
                "approved": False,
                "evaluation": evaluation,
                "error": "Could not parse a verdict from the safety review",
                "model": model
            }

        verdict = parsed.get("approved")
        approved = verdict is True

        return {
            "success": True,
            "proposed_action": proposed_action,
            "approved": approved,
            "reasoning": parsed.get("reasoning"),
            "concerns": parsed.get("concerns", []),
            "suggestions": parsed.get("suggestions", []),
            "evaluation": evaluation,
            "model": model
        }

    except Exception as e:
        return {"success": False, "error": f"Guarding failed: {str(e)}"}

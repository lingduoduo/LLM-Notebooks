# pip install openai

import os
import json
from openai import OpenAI

client = OpenAI(api_key="OpenAI-API-Key")

### 结构化 Prompt（事前约束）
PROMPT1 = """
# Role
You are a sentiment analysis expert, skilled at identifying emotions from user feedback.

# Task
Please determine the sentiment of the user feedback. Follow the steps below to reason, then output the result. Only return "positive", "negative", or "neutral":

# Analysis Steps
1. **Keyword Extraction**: Identify all words with sentiment polarity in the sentence and label each word as positive, negative, or neutral.

2. **Context Analysis**:
   - Analyze logical relationships between words (contrast, progression, parallelism, sarcasm, etc.) and determine the core sentiment (e.g., in contrast sentences, the latter part is usually dominant; sarcasm should be interpreted based on context).
   - Identify neutral scenarios: If there are no clear positive/negative words, or if positive and negative signals are balanced and cancel each other out, classify as neutral.
   - Distinguish sentiment in internet slang (e.g., "666" indicates approval <positive>, "6" indicates sarcasm <negative>, "not bad" indicates neutral).

3. **Sentiment Weighting**: Rank extracted sentiment words by importance. Core sentiment expressions carry higher weight than secondary descriptions. Neutral words do not participate in weighting unless no clear sentiment exists or weights are equal.

4. **Final Conclusion**:
   - If positive weight > negative weight, return "positive";
   - If negative weight > positive weight, return "negative";
   - If equal or no clear sentiment, return "neutral".

# Input Format
{
  "feedback": "<user feedback content>"
}

# Output Format
{
  "sentiment": "positive/negative/neutral",
  "reason": "<reasoning>"
}

# Constraints
1. Must return JSON only, no additional text.
2. sentiment must be one of "positive", "negative", or "neutral".
3. reason must explain the decision using keyword extraction and context analysis. For neutral cases, explicitly state "no clear sentiment" or "objective description".
"""


### Few Shot Prompt（事前约束）
PROMPT2 = """
# Role
You are a sentiment analysis expert, skilled at identifying emotions from user feedback.

# Task
Please determine the sentiment of the user feedback. Follow the steps below to reason step by step before outputting the result. Only return "positive", "negative", or "neutral":

# Analysis Steps
1. **Keyword Extraction**: Identify all words in the sentence that carry sentiment and label each word with its polarity (positive/negative/neutral).

2. **Context Analysis**:
   - Analyze logical relationships between words (contrast, progression, parallelism, sarcasm, etc.) to determine the core sentiment (e.g., in contrast sentences, the latter part is usually dominant; sarcasm should be interpreted based on context).
   - Identify neutral scenarios: If there are no clear positive/negative words and the feedback is purely objective, or if positive and negative signals are balanced and cancel each other out, classify it as neutral.
   - Distinguish sentiment in internet slang (e.g., "666" indicates approval <positive>, "6" indicates sarcasm <negative>, "not bad" indicates neutral).

3. **Sentiment Weighting**: Rank the extracted sentiment words by importance. Words reflecting the core sentiment carry higher weight than secondary descriptions. Neutral words do not participate in weighting unless no clear positive/negative signals exist or weights are equal.

4. **Final Conclusion**:
   - If positive signals > negative signals, return "positive";
   - If negative signals > positive signals, return "negative";
   - If positive and negative signals are equal, or there are no clear signals and the content is purely objective, return "neutral".

# Input Format
{
  "feedback": "<user feedback content>"
}

# Output Format
{
  "sentiment": "positive/negative/neutral",
  "reason": "<reasoning>"
}

# Constraints
1. Must return JSON data only, no additional explanatory text.
2. The sentiment field must be one of "positive", "negative", or "neutral".
3. The reason field must explain the judgment based on keyword extraction and context analysis. For neutral feedback, explicitly state "no clear sentiment" or "objective description" and include all key points.

# Examples

## Example 1 Input
{"feedback": "This app is amazing!"}

## Example 1 Output
{"sentiment": "positive", "reason": "The user uses the positive phrase 'amazing' to directly praise the app. The core sentiment is satisfaction, with no negative signals."}

## Example 2 Input
{"feedback": "The interface is nice but it's slow. It takes a minute to load every time, so I uninstalled it."}

## Example 2 Output
{"sentiment": "negative", "reason": "Extracted positive word 'nice' and negative words 'slow', 'takes a minute to load', and 'uninstalled'. The contrast 'but' makes the latter part dominant, and the negative descriptions and uninstall action reflect dissatisfaction. Negative signals outweigh positive signals."}

## Example 3 Input
{"feedback": "It's okay, nothing special."}

## Example 3 Output
{"sentiment": "neutral", "reason": "The phrase 'okay' indicates mild neutrality and 'nothing special' suggests no strong positive or negative sentiment. Overall, there is no clear dominant sentiment."}

## Example 4 Input
{"feedback": "Great features, but the battery drains too fast."}

## Example 4 Output
{"sentiment": "negative", "reason": "Positive phrase 'great features' is outweighed by negative phrase 'battery drains too fast'. The contrast 'but' emphasizes the latter, indicating dissatisfaction."}

## Example 5 Input
{"feedback": "Not bad, could be better."}

## Example 5 Output
{"sentiment": "neutral", "reason": "The phrase 'not bad' suggests mild positive sentiment, while 'could be better' introduces a mild negative aspect. The signals balance each other, resulting in a neutral sentiment."}
"""

def analyze_sentiment(feedback: str, prompt):
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {"feedback": feedback},
                    ensure_ascii=False
                )
            }
        ],
        temperature=0
    )

    return json.loads(response.output_text)


if __name__ == "__main__":
    feedback = "The interface is nice but very slow. I uninstalled it."
    for prompt in [PROMPT1, PROMPT2]:
        result = analyze_sentiment(feedback, prompt)
        print(json.dumps(result, indent=2))

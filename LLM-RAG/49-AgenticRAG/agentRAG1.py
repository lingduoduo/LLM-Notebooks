# pip install crewai docling

from crewai import LLM, Agent, Crew, Process, Task
from crewai.knowledge.source.crew_docling_source import CrewDoclingSource
import dotenv
import os

dotenv.load_dotenv(override=True)
print(os.environ["OPENAI_API_KEY"])

# Create a knowledge source from web content
content_source = CrewDoclingSource(
    file_paths=[
        "https://developer.volcengine.com/articles/7373874019113631754",
        "https://docs.crewai.com/en/concepts/knowledge"
    ],
)

# Create an LLM with temperature=0 to ensure deterministic output
llm = LLM(model="gpt-4o-mini", temperature=0, api_key=os.environ["OPENAI_API_KEY"])

# Create an agent using the knowledge base
agent = Agent(
    role="Article Comprehension Expert",
    goal="Read and fully understand all contents of the articles.",
    backstory="You are an expert at understanding articles and their content.",
    verbose=True,
    allow_delegation=False,
    llm=llm,
)

# Define the task for the agent
task = Task(
    description="Answer the following question about the articles: {question}",
    expected_output="An answer to the question.",
    agent=agent,
)

# Create the crew
crew = Crew(
    agents=[agent],
    tasks=[task],
    verbose=True,
    process=Process.sequential,
    knowledge_sources=[content_source],
)

# Execute the workflow
result = crew.kickoff(
    inputs={"question": "What is Agentic RAG? Please be sure to provide sources."}
)

'''
(llm_clean)  ✘  🐍 llm_clean  linghuang@Mac  ~/Git/LLMs   rag-optimization  /Users/linghuang/miniconda3/envs/llm_clean/bin/python /Users/linghuang/Git/LLMs/LLM-RAG/49-AgenticRAG/agent
RAG1.py
╭──────────────────────────────────────────────────────────────────────────────── 🚀 Crew Execution Started ────────────────────────────────────────────────────────────────────────────────╮
│                                                                                                                                                                                           │
│  Crew Execution Started                                                                                                                                                                   │
│  Name:                                                                                                                                                                                    │
│  crew                                                                                                                                                                                     │
│  ID:                                                                                                                                                                                      │
│  d5c3cd4b-afb7-43ab-808a-2e45e4aacbae                                                                                                                                                     │
│                                                                                                                                                                                           │
│                                                                                                                                                                                           │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

╭───────────────────────────────────────────────────────────────────────────────────── 📋 Task Started ─────────────────────────────────────────────────────────────────────────────────────╮
│                                                                                                                                                                                           │
│  Task Started                                                                                                                                                                             │
│  Name: Answer the following question about the articles: What is Agentic RAG? Please be sure to provide sources.                                                                          │
│  ID: a0b05e06-3b3c-4902-b930-4cd9f3eb1ad7                                                                                                                                                 │
│                                                                                                                                                                                           │
│                                                                                                                                                                                           │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

╭───────────────────────────────────────────────────────────────────────────────── 🔍 Knowledge Retrieval ──────────────────────────────────────────────────────────────────────────────────╮
│                                                                                                                                                                                           │
│  Knowledge Retrieval Started                                                                                                                                                              │
│  Status: Retrieving...                                                                                                                                                                    │
│                                                                                                                                                                                           │
│                                                                                                                                                                                           │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

╭───────────────────────────────────────────────────────────────────────────────── 📚 Knowledge Retrieved ──────────────────────────────────────────────────────────────────────────────────╮
│                                                                                                                                                                                           │
│  Search Query:                                                                                                                                                                            │
│  What is Agentic RAG? Provide detailed explanations and sources.                                                                                                                          │
│  Knowledge Retrieved:                                                                                                                                                                     │
│  Additional Information: 让我们来一步步实现简单的Agentic RAG。                                                                                                                            │
│  在这里的Agentic RAG架构中：                                                                                                                                                              │
│  这里介绍Agentic RAG方案： **一种基于AI Agent的方法，借助Agent的任务规划与工具能力，来协调完成对多文档的、多类型的问答需求。**                                                            │
│  既能提供RAG的基础查询能力，也能提供基于RAG之上更多样与复杂任务能力。概念架构如下：                                                                                                       │
│                                                                                                                                                                                           │
│                                                                                                                                                                                           │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

╭──────────────────────────────────────────────────────────────────────────────────── 🤖 Agent Started ─────────────────────────────────────────────────────────────────────────────────────╮
│                                                                                                                                                                                           │
│  Agent: Article Comprehension Expert                                                                                                                                                      │
│                                                                                                                                                                                           │
│  Task: Answer the following question about the articles: What is Agentic RAG? Please be sure to provide sources.                                                                          │
│                                                                                                                                                                                           │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

╭────────────────────────────────────────────────────────────────────────────────── ✅ Agent Final Answer ──────────────────────────────────────────────────────────────────────────────────╮
│                                                                                                                                                                                           │
│  Agent: Article Comprehension Expert                                                                                                                                                      │
│                                                                                                                                                                                           │
│  Final Answer:                                                                                                                                                                            │
│  Agentic RAG是一种基于AI Agent的方法，借助Agent的任务规划与工具能力，来协调完成对多文档的、多类型的问答需求。它既能提供RAG的基础查询能力，也能提供基于RAG之上更多样与复杂任务能力。       │
│                                                                                                                                                                                           │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

╭─────────────────────────────────────────────────────────────────────────────────── 📋 Task Completion ────────────────────────────────────────────────────────────────────────────────────╮
│                                                                                                                                                                                           │
│  Task Completed                                                                                                                                                                           │
│  Name:                                                                                                                                                                                    │
│  Answer the following question about the articles: What is Agentic RAG? Please be sure to provide sources.                                                                                │
│  Agent:                                                                                                                                                                                   │
│  Article Comprehension Expert                                                                                                                                                             │
│                                                                                                                                                                                           │
│                                                                                                                                                                                           │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

╭───────────────────────────────────────────────────────────────────────────────────── Crew Completion ─────────────────────────────────────────────────────────────────────────────────────╮
│                                                                                                                                                                                           │
│  Crew Execution Completed                                                                                                                                                                 │
│  Name:                                                                                                                                                                                    │
│  crew                                                                                                                                                                                     │
│  ID:                                                                                                                                                                                      │
│  d5c3cd4b-afb7-43ab-808a-2e45e4aacbae                                                                                                                                                     │
│  Final Output: Agentic RAG是一种基于AI                                                                                                                                                    │
│  Agent的方法，借助Agent的任务规划与工具能力，来协调完成对多文档的、多类型的问答需求。它既能提供RAG的基础查询能力，也能提供基于RAG之上更多样与复杂任务能力。                               │
│                                                                                                                                                                                           │
│                                                                                                                                                                                           │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
───────────────────────────────────────────────────────────────────────────────────── Tracing Status ──────────────────────────────────────────────────────────────────────────────────────╮
│                                                                                                                                                                                           │
│  Info: Tracing is disabled.                                                                                                                                                               │
│                                                                                                                                                                                           │
│  To enable tracing, do any one of these:                                                                                                                                                  │
│  • Set tracing=True in your Crew/Flow code                                                                                                                                                │
│  • Set CREWAI_TRACING_ENABLED=true in your project's .env file                                                                                                                            │
│  • Run: crewai traces enable                                                                                                                                                              │
│                                                                                                                                                                                           │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
                                                                                                                                                      │
'''
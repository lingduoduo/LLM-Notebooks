import json
import os
from typing import Dict, List, Optional, TypedDict, NotRequired

from openai import OpenAI
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings


# =============================================================================
# Environment & Client
# =============================================================================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError(
        "Missing OPENAI_API_KEY. Please set it in your environment or in a .env file."
    )

client = OpenAI(api_key=OPENAI_API_KEY)


# =============================================================================
# OpenAI Embeddings Implementation
# =============================================================================
class OpenAIEmbeddings(Embeddings):
    """LangChain-compatible embeddings wrapper using OpenAI embeddings API."""

    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self.client = client

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [data.embedding for data in response.data]

    def embed_query(self, text: str) -> List[float]:
        response = self.client.embeddings.create(model=self.model, input=[text])
        return response.data[0].embedding


# =============================================================================
# State Definition (LangGraph)
# =============================================================================
class GenerationState(TypedDict):
    """
    Workflow state definition used to pass data between LangGraph nodes.

    Fields:
      - original_text: The raw input document
      - chunks: Paragraph/semantic chunks derived from the input
      - summaries: Short summaries for each chunk
      - planning_tree: A JSON-like outline containing a title and section list
      - final_output: The final generated report
      - vectorstore: FAISS vector store holding summaries and generated sections
    """

    original_text: str
    chunks: NotRequired[List[str]]
    summaries: NotRequired[List[str]]
    planning_tree: NotRequired[Dict]
    final_output: NotRequired[str]
    vectorstore: NotRequired[Optional[FAISS]]


# =============================================================================
# Model Initialization
# =============================================================================
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


# =============================================================================
# Core Utility Functions
# =============================================================================
def split_text(text: str) -> List[str]:
    """
    Semantic text chunking based on paragraph structure and semantic coherence.

    Guarantees (when possible) between 2 and 10 chunks.
    NOTE: This uses a hard-coded heuristic strategy for demonstration.
    """
    # Split by paragraphs and keep non-empty ones
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # If paragraph count is already in target range, return directly
    if 2 <= len(paragraphs) <= 10:
        return paragraphs

    # If too few paragraphs, split further by sentences (Chinese punctuation)
    if len(paragraphs) < 2:
        import re

        sentences: List[str] = []
        for para in paragraphs:
            sent_list = re.split(r"[。！？]", para)
            sentences.extend([s.strip() for s in sent_list if s.strip()])

        # Recombine sentences into ~3 chunks (2–4) when possible
        if len(sentences) >= 4:
            chunk_size = max(1, len(sentences) // 3)
            chunks: List[str] = []
            for i in range(0, len(sentences), chunk_size):
                chunk = "。".join(sentences[i : i + chunk_size]).strip()
                if chunk:
                    chunks.append(chunk + "。")
            return chunks[:10]
        else:
            return sentences[:10]

    # If too many paragraphs, merge adjacent ones to target ~8 chunks
    if len(paragraphs) > 10:
        chunk_size = max(1, len(paragraphs) // 8)
        chunks: List[str] = []
        for i in range(0, len(paragraphs), chunk_size):
            chunk_paras = paragraphs[i : i + chunk_size]
            chunks.append("\n\n".join(chunk_paras))
        return chunks[:10]

    return paragraphs


def generate_summary(chunk: str) -> str:
    """
    Generate a concise summary, ensuring the length is <= 30% of the original chunk.
    """
    chunk_length = len(chunk)
    target_length = int(chunk_length * 0.3)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Please produce a highly concise summary of the following content.\n"
                    f"Requirements:\n"
                    f"1. Summary length must not exceed {target_length} characters (~30% of original)\n"
                    "2. Keep only the most essential ideas and key information\n"
                    "3. Use concise language and avoid redundancy\n"
                    "4. Keep the logic clear and highlight the key points"
                ),
            },
            {"role": "user", "content": chunk},
        ],
        temperature=0,
    )

    summary = response.choices[0].message.content.strip()

    # If still too long, compress again
    if len(summary) > target_length:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"Further compress the following summary to within {target_length} characters, "
                        "keeping only the most critical information:"
                    ),
                },
                {"role": "user", "content": summary},
            ],
            temperature=0,
        )
        summary = response.choices[0].message.content.strip()

    return summary


def build_planning_tree(summaries: List[str]) -> Dict:
    """
    Build a compact report outline (3–4 main sections) from chunk summaries.
    Output is strict JSON (or falls back to a default structure).
    """
    combined = "\n\n".join(f"Block {i+1}: {s}" for i, s in enumerate(summaries))
    prompt = f"""
Based on the following summarized text blocks, generate a concise report outline.

Objective:
- Analyze the summaries and produce a logically coherent structure.

Requirements:
- Generate only 3–4 main sections total.
- Each section should be a single consolidated paragraph (no sub-sections).
- Merge related ideas into integrated sections.
- Output must be strict JSON only; do not include any other text.

Summary input:
{combined}

Output format (note: subsections must be empty arrays):
{{
  "title": "Main Report Title",
  "sections": [
    {{"title": "Current State and Technical Foundations", "subsections": []}},
    {{"title": "Applications and Practical Use Cases", "subsections": []}},
    {{"title": "Challenges and Future Trends", "subsections": []}}
  ]
}}
""".strip()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    content = response.choices[0].message.content.strip()

    # Remove possible markdown fences
    if content.startswith("```json"):
        content = content[7:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    # Parse JSON; fall back if parsing fails
    try:
        if content.startswith("{"):
            return json.loads(content)
    except Exception:
        pass

    return {
        "title": "Document Analysis Report",
        "sections": [
            {"title": "Core Technologies and Current State", "subsections": []},
            {"title": "Applications and Industry Impact", "subsections": []},
            {"title": "Challenges and Future Outlook", "subsections": []},
        ],
    }


def retrieve_relevant_memory(query: str, vectorstore: Optional[FAISS], k: int = 3) -> str:
    """Retrieve top-k relevant stored texts from the FAISS vector store."""
    if vectorstore is None:
        return "Vector store unavailable."
    docs = vectorstore.similarity_search(query, k=k)
    return "\n".join(d.page_content for d in docs)


def generate_section_content(title: str, context: str) -> str:
    """
    Generate a consolidated section paragraph based on retrieved context.
    """
    prompt = f"""
You are a professional writer. Based on the reference context, write a consolidated section.

# Reference context:
{context}

# Target section:
{title}

Requirements:
1. Merge all relevant content into a single cohesive paragraph.
2. Cover the core ideas and key information.
3. Be concise, logically structured, and avoid redundancy.
4. Paragraph length: 200–400 words.
5. Demonstrate analytical depth and professional insight.
""".strip()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


# =============================================================================
# Workflow Node Definitions
# =============================================================================
def split_node(state: GenerationState) -> GenerationState:
    print("=" * 60)
    print("🔄 [Chunking Stage] Starting text segmentation")
    print("=" * 60)

    chunks = split_text(state["original_text"])
    state["chunks"] = chunks

    print("📊 Chunking statistics:")
    print(f"   Original text length: {len(state['original_text'])} characters")
    print(f"   Number of chunks: {len(chunks)}")
    avg_length = sum(len(chunk) for chunk in chunks) // len(chunks) if chunks else 0
    print(f"   Average chunk length: {avg_length} characters")

    print("\n📝 Chunk details:")
    for i, chunk in enumerate(chunks, 1):
        words = len(chunk.split())
        chars = len(chunk)
        preview = chunk[:50] + "..." if len(chunk) > 50 else chunk
        print(f"   Chunk {i}: {words} words ({chars} chars) | {preview}")

    chunk_sizes = [len(chunk) for chunk in chunks] if chunks else [0]
    size_variance = max(chunk_sizes) - min(chunk_sizes)
    print("\n📏 Chunk balance analysis:")
    print(f"   Largest chunk: {max(chunk_sizes)} chars")
    print(f"   Smallest chunk: {min(chunk_sizes)} chars")
    print(f"   Size variance: {size_variance} chars")

    print("✅ Chunking stage completed\n")
    return state


def summarize_and_memorize_node(state: GenerationState) -> GenerationState:
    """
    Purpose:
    - Summarize each chunk
    - Store summaries in state["summaries"]
    - Build a FAISS vector store from summaries
    - Print key diagnostics
    """
    print("=" * 60)
    print("🧠 [Memory Stage] Building contextual memory")
    print("=" * 60)

    chunks = state.get("chunks", [])
    summaries: List[str] = []

    print("📝 Generating summaries...")
    for i, chunk in enumerate(chunks, 1):
        print(f"   Processing chunk {i}/{len(chunks)}...", end=" ")
        summary = generate_summary(chunk)
        summaries.append(summary)

        compression_ratio = (len(summary) / max(len(chunk), 1)) * 100
        print("✅")
        print(f"      Original: {len(chunk)} chars")
        print(f"      Summary:  {len(summary)} chars (compression: {compression_ratio:.1f}%)")
        print(f"      Preview:  {summary[:60]}...")

    state["summaries"] = summaries

    print("\n🔍 Building vector store...")
    state["vectorstore"] = FAISS.from_texts(summaries, embedding=embeddings)
    print("✅ Vector store built")

    print("📊 Vector store stats:")
    print(f"   Documents stored: {len(summaries)}")
    print("   Embedding dim: 1536 (text-embedding-3-small)")

    print("\n🔑 Quick keyword index (naive):")
    for i, summary in enumerate(summaries, 1):
        keywords = [w for w in summary.split()[:5] if len(w) > 2]
        print(f"   Doc {i}: {', '.join(keywords)}")

    print("✅ Memory stage completed\n")
    return state


def planning_node(state: GenerationState) -> GenerationState:
    """
    Planning stage:
    - Build a compact planning tree from summaries
    - Print the outline
    """
    print("=" * 60)
    print("📋 [Planning Stage] Building a compact report outline")
    print("=" * 60)

    print("🤖 Analyzing summaries and generating outline...")
    planning_tree = build_planning_tree(state.get("summaries", []))
    state["planning_tree"] = planning_tree

    print("✅ Outline generated")
    print("\n📖 Report outline:")
    print(f"   Title: {planning_tree.get('title', 'Undefined')}")

    sections = planning_tree.get("sections", [])
    print(f"   Number of main sections: {len(sections)}")

    for i, section in enumerate(sections, 1):
        section_title = section.get("title", f"Section {i}")
        subsections = section.get("subsections", [])
        print(f"   {i}. {section_title}")
        if subsections:
            for j, subsection in enumerate(subsections, 1):
                print(f"      {i}.{j} {subsection}")
        else:
            print("      (Single consolidated paragraph; no subsections)")

    print("\n🎯 Generation strategy:")
    content_paragraphs = len(sections)
    print(f"   Expected paragraphs: {content_paragraphs}")
    print("   Strategy: one consolidated paragraph per section")
    print("   ✅ Target range: 3–5 paragraphs")

    print("✅ Planning stage completed\n")
    return state


def generate_node(state: GenerationState) -> GenerationState:
    """
    Generation stage:
    - Generate consolidated content for each planned section
    - Append generated content into final output
    - Optionally add generated content back into vector store (memory)
    """
    print("=" * 60)
    print("✍️ [Generation Stage] Generating consolidated sections")
    print("=" * 60)

    tree = state.get("planning_tree", {})
    content_parts: List[str] = []

    if "title" in tree:
        title = tree["title"]
        content_parts.append(f"# {title}\n")
        print(f"📝 Main title generated: {title}")

    sections = tree.get("sections", [])
    print(f"🎯 Using compact strategy: {len(sections)} main sections (no subsections)")

    for i, section in enumerate(sections, 1):
        sec_title = section.get("title", f"Section {i}")

        print(f"\n🔄 Generating section {i}/{len(sections)}: {sec_title}")
        content_parts.append(f"## {sec_title}")

        context = retrieve_relevant_memory(sec_title, state.get("vectorstore"))
        print(f"   📚 Retrieved context length: {len(context)} chars")

        content = generate_section_content(sec_title, context)
        content_parts.append(content)
        print(f"   ✅ Section content generated: {len(content)} chars")

        if state.get("vectorstore") is not None:
            state["vectorstore"].add_texts([content])
            print("   💾 Added section content to memory store")

    state["final_output"] = "\n\n".join(content_parts)

    content_paragraphs = len([p for p in content_parts if not p.startswith("#")])
    print("\n📊 Generation summary:")
    print(f"   Total chars: {len(state['final_output'])}")
    print(f"   Main sections: {len(sections)}")
    print(f"   Content paragraphs: {content_paragraphs}")
    print(f"   Total parts: {len(content_parts)}")
    print(f"   ✅ Controlled paragraph count to ~{content_paragraphs} (target: 3–5)")
    print("✅ Generation stage completed\n")

    return state


# =============================================================================
# Build the StateGraph
# =============================================================================
def create_generation_workflow() -> StateGraph:
    """
    Layer 1: Original text -> chunk-level summaries (local compression)
    Layer 2: Summaries -> outline planning (global organization)
    Layer 3: Use the outline + retrieved memory -> regenerate (deep synthesis)
    """
    workflow = StateGraph(GenerationState)

    workflow.add_node("split", split_node)
    workflow.add_node("summarize_and_memorize", summarize_and_memorize_node)
    workflow.add_node("plan", planning_node)
    workflow.add_node("generate", generate_node)

    workflow.set_entry_point("split")
    workflow.add_edge("split", "summarize_and_memorize")
    workflow.add_edge("summarize_and_memorize", "plan")
    workflow.add_edge("plan", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


# =============================================================================
# Execution Example
# =============================================================================
if __name__ == "__main__":
    print("Starting the long-form text generation system")
    print("=" * 60)

    sample_text = """
IN THE OCEAN OF LONELINESS AND DIGNITY: RE-READING THE OLD MAN AND THE SEA

At dawn in Havana, an old fisherman named Santiago rows alone toward the open sea. This seemingly ordinary scene
forms the entire narrative space of Hemingway’s The Old Man and the Sea. Published in 1952, this novella—known for
its concise, forceful prose and profound symbolism—became an undeniable classic in 20th-century literature.
Revisiting the work across the haze of more than half a century, we discover that The Old Man and the Sea is far more
than a simple tale of “an old man catching a fish.” It is an enduring hymn to the power of the human spirit, a deep
interrogation of existence itself, and a mirror reflecting modern spiritual dilemmas.

Hemingway’s portrayal of Santiago reveals a sharp understanding of the human inner world. After eighty-four days
without a catch, the old Cuban fisherman stands on the edge of society in material terms: his harvest is meager, his
boat is worn, his tools are crude, and other fishermen consider him a failure. Only a boy named Manolin continues to
respect him sincerely. Yet it is precisely in this “failed” old man that Hemingway locates the noblest qualities of the
human spirit—dignity and resilience.
""".strip()

    workflow_app = create_generation_workflow()

    initial_state: GenerationState = {
        "original_text": sample_text,
        "chunks": [],
        "summaries": [],
        "planning_tree": {},
        "final_output": "",
        "vectorstore": None,
    }

    result = workflow_app.invoke(initial_state)

    print("=" * 60)
    print("Completed!")
    print("=" * 60)
    print("\n📄 Final generated output:")
    print("-" * 40)
    print(result["final_output"])



'''
(llm_clean)  🐍 llm_clean  linghuang@Mac  ~/Git/LLMs/LLM-RAG/22/22   rag-optimization  /Users/linghuang/miniconda3/envs/llm_clean/bin/python /Users/linghuang/Git/LLMs/
LLM-RAG/23/long_prompt.py
Starting the long-form text generation system
============================================================
============================================================
🔄 [Chunking Stage] Starting text segmentation
============================================================
📊 Chunking statistics:
   Original text length: 1236 characters
   Number of chunks: 3
   Average chunk length: 410 characters

📝 Chunk details:
   Chunk 1: 14 words (74 chars) | IN THE OCEAN OF LONELINESS AND DIGNITY: RE-READING...
   Chunk 2: 111 words (655 chars) | At dawn in Havana, an old fisherman named Santiago...
   Chunk 3: 82 words (503 chars) | Hemingway’s portrayal of Santiago reveals a sharp ...

📏 Chunk balance analysis:
   Largest chunk: 655 chars
   Smallest chunk: 74 chars
   Size variance: 581 chars
✅ Chunking stage completed

============================================================
🧠 [Memory Stage] Building contextual memory
============================================================
📝 Generating summaries...
   Processing chunk 1/3... ✅
      Original: 74 chars
      Summary:  28 chars (compression: 37.8%)
      Preview:  Themes: loneliness, dignity....
   Processing chunk 2/3... ✅
      Original: 655 chars
      Summary:  195 chars (compression: 29.8%)
      Preview:  Hemingway's 1952 novella, The Old Man and the Sea, follows S...
   Processing chunk 3/3... ✅
      Original: 503 chars
      Summary:  132 chars (compression: 26.2%)
      Preview:  Hemingway's Santiago embodies dignity and resilience despite...

🔍 Building vector store...
✅ Vector store built
📊 Vector store stats:
   Documents stored: 3
   Embedding dim: 1536 (text-embedding-3-small)

🔑 Quick keyword index (naive):
   Doc 1: Themes:, loneliness,, dignity.
   Doc 2: Hemingway's, 1952, novella,, The, Old
   Doc 3: Hemingway's, Santiago, embodies, dignity, and
✅ Memory stage completed

============================================================
📋 [Planning Stage] Building a compact report outline
============================================================
🤖 Analyzing summaries and generating outline...
✅ Outline generated

📖 Report outline:
   Title: Analysis of Themes in Hemingway's The Old Man and the Sea
   Number of main sections: 3
   1. Exploration of Loneliness and Dignity
      (Single consolidated paragraph; no subsections)
   2. Santiago as a Symbol of the Human Spirit
      (Single consolidated paragraph; no subsections)
   3. Recognition of Worth in a Failing Society
      (Single consolidated paragraph; no subsections)

🎯 Generation strategy:
   Expected paragraphs: 3
   Strategy: one consolidated paragraph per section
   ✅ Target range: 3–5 paragraphs
✅ Planning stage completed

============================================================
✍️ [Generation Stage] Generating consolidated sections
============================================================
📝 Main title generated: Analysis of Themes in Hemingway's The Old Man and the Sea
🎯 Using compact strategy: 3 main sections (no subsections)

🔄 Generating section 1/3: Exploration of Loneliness and Dignity
   📚 Retrieved context length: 357 chars
   ✅ Section content generated: 1686 chars
   💾 Added section content to memory store

🔄 Generating section 2/3: Santiago as a Symbol of the Human Spirit
   📚 Retrieved context length: 2015 chars
   ✅ Section content generated: 1790 chars
   💾 Added section content to memory store

🔄 Generating section 3/3: Recognition of Worth in a Failing Society
   📚 Retrieved context length: 1952 chars
   ✅ Section content generated: 1795 chars
   💾 Added section content to memory store

📊 Generation summary:
   Total chars: 5470
   Main sections: 3
   Content paragraphs: 3
   Total parts: 7
   ✅ Controlled paragraph count to ~3 (target: 3–5)
✅ Generation stage completed

============================================================
Completed!
============================================================

📄 Final generated output:
----------------------------------------
# Analysis of Themes in Hemingway's The Old Man and the Sea


## Exploration of Loneliness and Dignity

In Hemingway's 1952 novella, *The Old Man and the Sea*, the themes of loneliness and dignity are intricately woven into the fabric of Santiago's character, an old fisherman who embodies the resilience of the human spirit amidst societal neglect. Santiago's solitary existence on the vast, unforgiving sea serves as a poignant backdrop for his existential struggles, highlighting the profound isolation that often accompanies aging and failure. Despite the relentless challenges he faces, including a long streak of bad luck that leaves him marginalized by the fishing community, Santiago maintains an unwavering sense of dignity. This dignity is not merely a facade; it is a testament to his inner strength and unwavering commitment to his craft. The only person who truly recognizes his worth is the young boy, Manolin, whose loyalty and admiration stand in stark contrast to the indifference of the broader society. Their relationship underscores the theme of connection amidst loneliness, as Manolin’s presence offers Santiago a glimmer of hope and companionship. Through Santiago's journey, Hemingway transcends the narrative of a simple fishing tale, delving into the complexities of human existence, where dignity is often forged in the crucible of solitude. Ultimately, Santiago's struggle against the marlin becomes a metaphor for the broader human condition, illustrating that even in the face of overwhelming odds and societal failure, the spirit can endure, and dignity can prevail. This exploration of loneliness and dignity not only enriches the narrative but also invites readers to reflect on their own lives and the inherent value of resilience in the face of adversity.

## Santiago as a Symbol of the Human Spirit

In Hemingway's *The Old Man and the Sea*, Santiago emerges as a profound symbol of the human spirit, embodying dignity and resilience in the face of societal neglect and personal adversity. The old fisherman’s solitary existence on the vast, unforgiving sea serves as a poignant backdrop for his existential struggles, illustrating the profound isolation that often accompanies aging and failure. Despite enduring a long streak of bad luck that marginalizes him within the fishing community, Santiago clings to an unwavering sense of dignity, which reflects his inner strength and steadfast commitment to his craft. This dignity is not merely a facade; it is a testament to his character, forged in the crucible of solitude. The only individual who truly recognizes his worth is the young boy, Manolin, whose loyalty and admiration starkly contrast with the indifference of the broader society. Their relationship highlights the theme of connection amidst loneliness, as Manolin’s presence offers Santiago a glimmer of hope and companionship, reinforcing the notion that human connections can provide solace even in the darkest of times. Santiago's epic struggle against the marlin transcends the narrative of a simple fishing tale, evolving into a metaphor for the broader human condition. It illustrates that, even when faced with overwhelming odds and societal failure, the spirit can endure, and dignity can prevail. Through this exploration of loneliness and dignity, Hemingway invites readers to reflect on their own lives, emphasizing the inherent value of resilience in the face of adversity. Ultimately, Santiago stands as a testament to the enduring strength of the human spirit, reminding us that true worth is often recognized only by those who choose to see beyond the surface.

## Recognition of Worth in a Failing Society

In Hemingway's *The Old Man and the Sea*, Santiago serves as a profound symbol of dignity and resilience amidst societal neglect and personal adversity. His solitary existence on the vast, unforgiving sea underscores the profound isolation that often accompanies aging and failure, particularly as he endures a long streak of bad luck that marginalizes him within the fishing community. Despite this, Santiago clings to an unwavering sense of dignity, a testament to his inner strength and steadfast commitment to his craft. This dignity is not merely a facade; it is forged in the crucible of solitude, reflecting the essence of his character. The only individual who truly recognizes his worth is the young boy, Manolin, whose loyalty and admiration starkly contrast with the indifference of the broader society. Their relationship highlights the theme of connection amidst loneliness, as Manolin’s presence offers Santiago a glimmer of hope and companionship, reinforcing the notion that human connections can provide solace even in the darkest of times. Santiago's epic struggle against the marlin transcends the narrative of a simple fishing tale, evolving into a metaphor for the broader human condition. It illustrates that, even when faced with overwhelming odds and societal failure, the spirit can endure, and dignity can prevail. Through this exploration of loneliness and dignity, Hemingway invites readers to reflect on their own lives, emphasizing the inherent value of resilience in the face of adversity. Ultimately, Santiago stands as a testament to the enduring strength of the human spirit, reminding us that true worth is often recognized only by those who choose to see beyond the surface, illuminating the profound impact of recognition and connection in a failing society.
'''
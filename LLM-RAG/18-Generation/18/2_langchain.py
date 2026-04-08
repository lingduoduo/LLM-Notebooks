import os
import re
from dataclasses import dataclass
from datetime import datetime
from textwrap import dedent
from typing import List, Optional, Sequence, Tuple

import dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

dotenv.load_dotenv()


@dataclass(frozen=True, slots=True)
class SourceInfo:
    """Information about a data source."""

    url: str
    title: str
    content: str
    timestamp: datetime


class SimpleRAGSystem:
    """A small LangChain RAG demo with source display."""

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 3,
    ) -> None:
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set. Please set it as an environment variable.")

        self.top_k = top_k
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
        self.llm = OpenAI(
            model="gpt-3.5-turbo-instruct",
            openai_api_key=self.api_key,
            temperature=0,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        self.vectorstore: Optional[FAISS] = None
        self.answer_chain = self._build_answer_chain()

    @staticmethod
    def _preprocess_text(text: str) -> str:
        """Normalize text with lightweight regex cleaning before indexing or prompting."""
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        normalized = re.sub(r"[ \t]+", " ", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        normalized = re.sub(r"[^\S\n]+([,.;:!?])", r"\1", normalized)
        return normalized.strip()

    def _build_answer_chain(self):
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=dedent(
                """
                You are a fact-checking expert. Please answer the question based on the provided context.

                Context:
                {context}

                Question: {question}

                Please answer in the following format:
                Verification Result: [True/False/Uncertain]
                Confidence: [0-100%]
                Reasoning: [Explain your reasoning in detail]
                Evidence: [Quote the specific supporting evidence]
                """
            ).strip(),
        )
        return prompt_template | self.llm | StrOutputParser()

    def get_knowledge_sources(self) -> List[SourceInfo]:
        """Get knowledge sources (predefined content; no web crawling)."""
        now = datetime.now()
        return [
            SourceInfo(
                url="https://wikipedia.org/wiki/artificial_intelligence",
                title="Artificial Intelligence - Wikipedia",
                content=dedent(
                    """
                    Artificial intelligence (AI) is a branch of computer science focused on creating machines
                    and software that can perform tasks typically requiring human intelligence.

                    Major areas of AI include:
                    1. Machine learning
                    2. Deep learning
                    3. Natural language processing
                    4. Computer vision
                    5. Expert systems

                    There is significant debate about whether AI will surpass human intelligence within the next decade.
                    Some argue AGI may take decades; others believe breakthroughs could come sooner.
                    Current AI excels at specific tasks but remains far from general intelligence.
                    """
                ).strip(),
                timestamp=now,
            ),
            SourceInfo(
                url="https://wikipedia.org/wiki/machine_learning",
                title="Machine Learning - Wikipedia",
                content=dedent(
                    """
                    Machine learning is a branch of AI focused on algorithms that learn from data and improve over time.

                    Relationship:
                    - AI is broader
                    - Machine learning is one major approach to AI
                    - Deep learning is a subfield of machine learning

                    Hierarchy: AI > Machine Learning > Deep Learning

                    Types:
                    1. Supervised learning
                    2. Unsupervised learning
                    3. Reinforcement learning
                    """
                ).strip(),
                timestamp=now,
            ),
            SourceInfo(
                url="https://wikipedia.org/wiki/deep_learning",
                title="Deep Learning - Wikipedia",
                content=dedent(
                    """
                    Deep learning is a subfield of machine learning that uses multi-layer neural networks.

                    Hierarchy:
                    - AI is the broadest
                    - Machine learning is a branch of AI
                    - Deep learning is a specialized area of machine learning
                    """
                ).strip(),
                timestamp=now,
            ),
            SourceInfo(
                url="https://wikipedia.org/wiki/Python_(programming_language)",
                title="Python - Wikipedia",
                content=dedent(
                    """
                    Python is a high-level programming language known for its readable syntax.

                    Python in data science:
                    Python is one of the most popular languages in data science.
                    Surveys often report high usage among data scientists (e.g., Kaggle, Stack Overflow).
                    Python's advantages include a rich library ecosystem and strong community support.
                    """
                ).strip(),
                timestamp=now,
            ),
        ]

    def _preprocess_sources(self, sources: Sequence[SourceInfo]) -> List[SourceInfo]:
        return [
            SourceInfo(
                url=source.url,
                title=self._preprocess_text(source.title),
                content=self._preprocess_text(source.content),
                timestamp=source.timestamp,
            )
            for source in sources
        ]

    def _build_documents(self, sources: Sequence[SourceInfo]) -> List[Document]:
        base_documents = [
            Document(
                page_content=self._preprocess_text(source.content),
                metadata={
                    "source": source.url,
                    "title": self._preprocess_text(source.title),
                    "timestamp": source.timestamp.isoformat(),
                },
            )
            for source in sources
        ]
        return self.text_splitter.split_documents(base_documents)

    @staticmethod
    def _format_docs(docs: Sequence[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    def build_knowledge_base(self) -> None:
        """Build the FAISS knowledge base."""
        print("Building the knowledge base...")

        sources = self._preprocess_sources(self.get_knowledge_sources())
        documents = self._build_documents(sources)
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)

        print(f"Knowledge base built successfully, containing {len(documents)} document chunks")

    def _ensure_ready(self) -> None:
        if self.vectorstore is None:
            raise ValueError("Knowledge base not initialized. Call build_knowledge_base() first.")

    def answer_question(self, question: str) -> Tuple[str, List[Tuple[Document, float]]]:
        """Retrieve sources once, then generate an answer from the retrieved context."""
        self._ensure_ready()
        assert self.vectorstore is not None

        cleaned_question = self._preprocess_text(question)
        results = self.vectorstore.similarity_search_with_score(cleaned_question, k=self.top_k)
        docs = [doc for doc, _ in results]
        answer = self.answer_chain.invoke(
            {
                "context": self._preprocess_text(self._format_docs(docs)),
                "question": cleaned_question,
            }
        )
        return answer, results

    def ask_question(self, question: str) -> Tuple[str, List[Tuple[Document, float]]]:
        """Ask a question, print the answer and retrieved sources."""
        print(f"Question: {question}")
        print("Thinking...")

        answer, source_results = self.answer_question(question)

        print("AI Answer:")
        print(answer)

        if source_results:
            print("\nEvidence Sources:")
            for index, (doc, score) in enumerate(source_results, start=1):
                print(f"   {index}. {doc.metadata.get('title', 'Unknown Source')}")
                print(f"      Link: {doc.metadata.get('source', 'No Link')}")
                print(f"      Score: {score:.4f}")
                snippet = self._preprocess_text(doc.page_content)[:200]
                print(f"      Snippet: {snippet}...")
                print()

        return answer, source_results


def main() -> None:
    print("=== Simple LangChain (LCEL) RAG System Demo ===\n")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Please set the OPENAI_API_KEY environment variable")
        return

    print("Initializing RAG system...")
    rag_system = SimpleRAGSystem(openai_api_key=openai_api_key)
    rag_system.build_knowledge_base()

    questions = [
        "Will artificial intelligence surpass human intelligence within the next decade?",
        "Is deep learning a subfield of machine learning?",
        "Is Python the most popular language in data science?",
    ]

    print("\n=== Starting Q&A Demo ===\n")
    for index, question in enumerate(questions, start=1):
        print(f"{'=' * 60}")
        print(f"Question {index}")
        print(f"{'=' * 60}")
        try:
            rag_system.ask_question(question)
        except Exception as exc:
            print(f"Error: {exc}")
        print(f"{'=' * 60}\n")

    print("Demo completed!")


if __name__ == "__main__":
    main()

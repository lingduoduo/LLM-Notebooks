import os
import re
from dataclasses import dataclass
from datetime import datetime
from textwrap import dedent
from typing import Any, Dict, List, Optional, Pattern, Sequence, Tuple

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


class DomainNERExtractor:
    """Rule-based extractor for domain and internet-infrastructure entities."""

    DOMAIN_PATTERN: Pattern[str] = re.compile(r"\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b", re.IGNORECASE)
    URL_PATTERN: Pattern[str] = re.compile(r"\bhttps?://[^\s/$.?#].[^\s]*\b", re.IGNORECASE)
    EMAIL_PATTERN: Pattern[str] = re.compile(
        r"\b[a-zA-Z0-9._%+-]+@(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b",
        re.IGNORECASE,
    )
    IPV4_PATTERN: Pattern[str] = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

    DOMAIN_KEYWORDS: Dict[str, Sequence[str]] = {
        "REGISTRAR": ("registrar", "godaddy", "namecheap", "dynadot", "register"),
        "REGISTRY": ("registry", "verisign", "registry operator"),
        "DNS": ("dns", "a record", "aaaa record", "cname", "mx", "txt", "ns record", "zone file"),
        "SECURITY": ("dnssec", "whois privacy", "ssl", "tls", "https"),
        "LIFECYCLE": ("register", "renew", "transfer", "expire", "redemption", "whois"),
        "ICANN": ("icann", "iana", "rdap", "whois"),
    }

    COMMON_TLDS = {"com", "org", "net", "edu", "gov", "io", "ai", "co", "us", "uk", "dev", "app"}

    def extract(self, text: str) -> Dict[str, Any]:
        """Extract domain-related entities from text."""
        urls = sorted(set(self.URL_PATTERN.findall(text)))
        emails = sorted(set(self.EMAIL_PATTERN.findall(text)))
        ip_addresses = sorted(set(self.IPV4_PATTERN.findall(text)))
        raw_domains = sorted(set(self.DOMAIN_PATTERN.findall(text)))

        email_domains = {email.rsplit("@", maxsplit=1)[-1].lower() for email in emails}
        domains = [domain for domain in raw_domains if domain.lower() not in email_domains]

        tlds = sorted(
            {
                domain.lower().rsplit(".", maxsplit=1)[-1]
                for domain in raw_domains
                if "." in domain and domain.lower().rsplit(".", maxsplit=1)[-1] in self.COMMON_TLDS
            }
        )

        lower_text = text.lower()
        keyword_entities = {
            label: [keyword for keyword in keywords if keyword in lower_text]
            for label, keywords in self.DOMAIN_KEYWORDS.items()
            if any(keyword in lower_text for keyword in keywords)
        }

        return {
            "domains": domains,
            "urls": urls,
            "emails": emails,
            "ip_addresses": ip_addresses,
            "tlds": tlds,
            "keyword_entities": keyword_entities,
        }


class DomainSearchRAGSystem:
    """A small LangChain RAG demo for domain search and NER."""

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 4,
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
        self.ner_extractor = DomainNERExtractor()

        self.vectorstore: Optional[FAISS] = None
        self.answer_chain = self._build_answer_chain()

    def _build_answer_chain(self):
        prompt_template = PromptTemplate(
            input_variables=["context", "question", "entities"],
            template=dedent(
                """
                You are a domain search and internet infrastructure expert.

                Answer the user's question using:
                1. The provided context
                2. The extracted domain-related entities

                Context:
                {context}

                Extracted Entities:
                {entities}

                Question:
                {question}

                Please answer in the following format:

                Answer: [Main answer]
                Entity Summary: [Summarize the important extracted entities]
                Reasoning: [Explain why the answer is supported by the context]
                Best Practice: [Provide one useful practical recommendation if applicable]
                """
            ).strip(),
        )
        return prompt_template | self.llm | StrOutputParser()

    def get_knowledge_sources(self) -> List[SourceInfo]:
        """Get knowledge sources for domain search, registration, and DNS."""
        now = datetime.now()
        return [
            SourceInfo(
                url="https://en.wikipedia.org/wiki/Domain_name",
                title="Domain Name - Wikipedia",
                content=dedent(
                    """
                    A domain name is a human-readable string used to identify resources on the Internet.
                    It maps to IP addresses through the Domain Name System (DNS).

                    A domain name typically includes:
                    1. A top-level domain (TLD), such as .com, .org, or .net
                    2. A second-level domain, such as wikipedia in wikipedia.org
                    3. Optional subdomains, such as www in www.example.com

                    Domain names make it easier for users to access online services without remembering numeric IP addresses.
                    """
                ).strip(),
                timestamp=now,
            ),
            SourceInfo(
                url="https://en.wikipedia.org/wiki/Domain_Name_System",
                title="Domain Name System - Wikipedia",
                content=dedent(
                    """
                    The Domain Name System (DNS) is a hierarchical and distributed naming system used to translate
                    human-readable domain names into IP addresses.

                    Important DNS record types include:
                    - A record: maps a domain to an IPv4 address
                    - AAAA record: maps a domain to an IPv6 address
                    - CNAME: alias for another domain name
                    - MX: mail exchange record for email routing
                    - NS: nameserver record
                    - TXT: arbitrary text, often used for verification and email security

                    DNS is essential for web browsing, email delivery, and many internet services.
                    """
                ).strip(),
                timestamp=now,
            ),
            SourceInfo(
                url="https://en.wikipedia.org/wiki/Domain_name_registrar",
                title="Domain Name Registrar - Wikipedia",
                content=dedent(
                    """
                    A domain name registrar is a company that manages the reservation of internet domain names.
                    Users register domain names through registrars.

                    A registrar typically allows users to:
                    - search whether a domain is available
                    - register a new domain
                    - renew an existing domain
                    - transfer a domain
                    - manage name server settings
                    - manage domain privacy or WHOIS-related settings

                    Registrars are often accredited and work with domain registries.
                    """
                ).strip(),
                timestamp=now,
            ),
            SourceInfo(
                url="https://en.wikipedia.org/wiki/Domain_name_registry",
                title="Domain Name Registry - Wikipedia",
                content=dedent(
                    """
                    A domain name registry is an organization that operates the database for a top-level domain (TLD).
                    For example, a registry may operate .com, .org, or country-code TLDs.

                    A registry is different from a registrar:
                    - The registry manages the TLD infrastructure
                    - The registrar interfaces with customers

                    Registries maintain the authoritative database of registered names under a TLD.
                    """
                ).strip(),
                timestamp=now,
            ),
            SourceInfo(
                url="https://en.wikipedia.org/wiki/ICANN",
                title="ICANN - Wikipedia",
                content=dedent(
                    """
                    ICANN is a nonprofit organization responsible for coordinating key parts of the internet's naming system.
                    It helps coordinate domain names, IP address allocation systems, and accreditation processes.

                    ICANN plays an important role in:
                    - registrar accreditation
                    - top-level domain policy
                    - DNS ecosystem coordination
                    - internet identifier stability
                    """
                ).strip(),
                timestamp=now,
            ),
            SourceInfo(
                url="https://en.wikipedia.org/wiki/Uniform_Resource_Locator",
                title="Uniform Resource Locator - Wikipedia",
                content=dedent(
                    """
                    A Uniform Resource Locator (URL) is a reference to a web resource that specifies its location.

                    A URL often includes:
                    - scheme, such as http or https
                    - host, such as www.example.com
                    - optional port
                    - path
                    - query parameters
                    - fragment

                    Example:
                    https://www.example.com/products?id=123

                    In this example:
                    - scheme: https
                    - host: www.example.com
                    - path: /products
                    - query: id=123
                    """
                ).strip(),
                timestamp=now,
            ),
            SourceInfo(
                url="https://en.wikipedia.org/wiki/WHOIS",
                title="WHOIS - Wikipedia",
                content=dedent(
                    """
                    WHOIS is a query and response protocol used for querying databases that store registered users
                    or assignees of internet resources, such as domain names.

                    WHOIS information may include:
                    - registrar name
                    - registration dates
                    - expiration dates
                    - name servers
                    - registrant or privacy service information

                    Some registration information may be hidden for privacy reasons.
                    """
                ).strip(),
                timestamp=now,
            ),
            SourceInfo(
                url="https://en.wikipedia.org/wiki/Top-level_domain",
                title="Top-Level Domain - Wikipedia",
                content=dedent(
                    """
                    A top-level domain (TLD) is the last segment of a domain name, appearing after the final dot.

                    Examples:
                    - .com
                    - .org
                    - .net
                    - .ai
                    - .io

                    TLDs can be generic or country-code based. Choice of TLD can affect branding, trust, and availability.
                    """
                ).strip(),
                timestamp=now,
            ),
        ]

    def _build_documents(self, sources: Sequence[SourceInfo]) -> List[Document]:
        base_documents = [
            Document(
                page_content=source.content,
                metadata={
                    "source": source.url,
                    "title": source.title,
                    "timestamp": source.timestamp.isoformat(),
                },
            )
            for source in sources
        ]
        return self.text_splitter.split_documents(base_documents)

    @staticmethod
    def _format_docs(docs: Sequence[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    @staticmethod
    def _format_entities(entities: Dict[str, Any]) -> str:
        return "\n".join(
            [
                f"Domains: {entities['domains']}",
                f"URLs: {entities['urls']}",
                f"Emails: {entities['emails']}",
                f"IP Addresses: {entities['ip_addresses']}",
                f"TLDs: {entities['tlds']}",
                f"Keyword Entities: {entities['keyword_entities']}",
            ]
        )

    def build_knowledge_base(self) -> None:
        """Build the FAISS knowledge base."""
        print("Building the domain knowledge base...")

        documents = self._build_documents(self.get_knowledge_sources())
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)

        print(f"Knowledge base built successfully with {len(documents)} document chunks.")

    def _ensure_ready(self) -> None:
        if self.vectorstore is None:
            raise ValueError("Knowledge base not initialized. Call build_knowledge_base() first.")

    def analyze_question(self, question: str) -> Dict[str, Any]:
        """Analyze a question with domain-oriented NER."""
        return self.ner_extractor.extract(question)

    def answer_question(
        self, question: str
    ) -> Tuple[str, Dict[str, Any], List[Tuple[Document, float]]]:
        """Run NER and retrieval once, then generate an answer."""
        self._ensure_ready()
        assert self.vectorstore is not None

        entities = self.analyze_question(question)
        results = self.vectorstore.similarity_search_with_score(question, k=self.top_k)
        docs = [doc for doc, _ in results]
        answer = self.answer_chain.invoke(
            {
                "context": self._format_docs(docs),
                "question": question,
                "entities": self._format_entities(entities),
            }
        )
        return answer, entities, results

    def ask_question(
        self, question: str
    ) -> Tuple[str, Dict[str, Any], List[Tuple[Document, float]]]:
        """Ask a question, print answer, extracted entities, and sources."""
        print(f"\nQuestion: {question}")
        print("Running domain NER...")

        answer, entities, source_results = self.answer_question(question)

        print("Extracted Entities:")
        for key, value in entities.items():
            print(f"  {key}: {value}")

        print("\nRAG Answer:")
        print(answer)

        if source_results:
            print("\nEvidence Sources:")
            for index, (doc, score) in enumerate(source_results, start=1):
                print(f"  {index}. {doc.metadata.get('title', 'Unknown Source')}")
                print(f"     URL: {doc.metadata.get('source', 'No Link')}")
                print(f"     Score: {score:.4f}")
                print(f"     Snippet: {doc.page_content[:220]}...")
                print()

        return answer, entities, source_results


def main() -> None:
    print("=== Domain Search + NER RAG System Demo ===\n")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Please set the OPENAI_API_KEY environment variable.")
        return

    print("Initializing domain RAG system...")
    rag_system = DomainSearchRAGSystem(openai_api_key=openai_api_key)
    rag_system.build_knowledge_base()

    questions = [
        "What is the difference between a registrar and a registry?",
        "How does DNS map example.com to an IP address?",
        "Can you extract entities from this text: Contact admin@example.com about https://openai.com and example.org using A record 192.168.1.1?",
        "What does the .ai TLD mean in a domain name?",
        "What information can WHOIS provide for a domain?",
    ]

    print("\n=== Starting Demo ===")
    for index, question in enumerate(questions, start=1):
        print("\n" + "=" * 70)
        print(f"Example Question {index}")
        print("=" * 70)
        try:
            rag_system.ask_question(question)
        except Exception as exc:
            print(f"Error: {exc}")

    print("\nDemo completed.")


if __name__ == "__main__":
    main()

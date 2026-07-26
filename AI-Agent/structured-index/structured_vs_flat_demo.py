"""
Structured index vs flat retrieval: offline comparison demo.

This module depends on no OpenAI account, no embedding model and no network --
plain Python plus networkx is enough. It uses a small hand-written knowledge
base about the Intel x86 SIMD instruction set to contrast two retrieval routes:

  * Flat retrieval: treat every fact as an independent chunk of text and recall
    it by lexical similarity. This is the abstraction of classic RAG
    ("split into chunks + vector search") -- it can only return isolated
    fragments.
  * Structured retrieval:
      - A GraphRAG-style entity/relationship graph: walking the relationship
        edges answers relational questions flat retrieval cannot, such as "what
        connects A to B" (the book's "multi-hop relational reasoning").
      - A RAPTOR-style hierarchical tree: detail is aggregated into higher-level
        summaries, which answers broad questions such as "give me an overview of
        this topic" that require synthesis across fragments (the book's
        "multi-level navigation").

This demo corresponds to the "comparative study of knowledge representation
philosophies" in experiment 3-8 (structured-index). Building a real index calls
an LLM (see `main.py build`); here the index contents are written out by hand so
you can see what structured indexing solves for flat retrieval without needing
an API key.
"""

import json
import re
from collections import deque
from typing import Dict, List, Optional, Tuple

import networkx as nx


# ---------------------------------------------------------------------------
# A small hand-written knowledge base (mirroring the Intel x86 sample document
# used in test_indexing.py). Each entity's description doubles as "one chunk of
# text" for the flat retriever.
# ---------------------------------------------------------------------------

ENTITIES: Dict[str, Dict[str, str]] = {
    "ADDPS": {"type": "instruction",
              "desc": "ADDPS: adds packed single-precision floating-point values in parallel, "
                      "processing four single-precision lanes at once."},
    "MOVAPS": {"type": "instruction",
               "desc": "MOVAPS: moves 128 bits of packed single-precision floating-point data "
                       "between a vector register and aligned memory."},
    "VADDPS": {"type": "instruction",
               "desc": "VADDPS: the AVX form of packed single-precision floating-point addition, "
                       "processing eight single-precision lanes at once."},
    "CPUID": {"type": "instruction",
              "desc": "CPUID: returns processor identification and feature information, used to "
                      "detect whether the processor supports extensions such as SSE and AVX."},
    "SSE": {"type": "extension",
            "desc": "SSE (Streaming SIMD Extensions): introduces 128-bit vector registers and "
                    "supports packed single-precision floating-point parallel arithmetic."},
    "AVX": {"type": "extension",
            "desc": "AVX (Advanced Vector Extensions): widens the vector registers to 256 bits, "
                    "further strengthening SIMD capability."},
    "XMM": {"type": "register",
            "desc": "XMM0-XMM15: 128-bit vector registers that hold packed data for SSE instructions."},
    "YMM": {"type": "register",
            "desc": "YMM0-YMM15: 256-bit vector registers used by AVX instructions; the low 128 bits "
                    "are shared with XMM."},
    "CR4.OSFXSR": {"type": "control-bit",
                   "desc": "CR4.OSFXSR: the control bit stating that the operating system supports "
                           "FXSAVE/FXRSTOR; it must be set to 1 before SSE instructions may be used."},
    "CR0.EM": {"type": "control-bit",
               "desc": "CR0.EM: the emulation flag; while it is 1 SIMD is disabled, so it must be "
                       "cleared before SSE or AVX instructions can execute."},
}

# Entity/relationship triples (subject, relation, object) forming the GraphRAG
# web of knowledge.
TRIPLES: List[Tuple[str, str, str]] = [
    ("ADDPS", "belongs_to", "SSE"),
    ("MOVAPS", "belongs_to", "SSE"),
    ("VADDPS", "belongs_to", "AVX"),
    ("ADDPS", "operates_on", "XMM"),
    ("VADDPS", "operates_on", "YMM"),
    ("SSE", "uses_register", "XMM"),
    ("AVX", "uses_register", "YMM"),
    ("AVX", "extends", "SSE"),
    ("SSE", "requires_enabling", "CR4.OSFXSR"),
    ("AVX", "requires_enabling", "CR4.OSFXSR"),
    ("SSE", "requires_cleared", "CR0.EM"),
    ("CPUID", "detects", "SSE"),
    ("CPUID", "detects", "AVX"),
]

# RAPTOR-style hierarchical tree: fine-grained leaves aggregated into a
# higher-level summary (the parent node).
TREE_SUMMARY = {
    "id": "Overview of the SIMD instruction set",
    "summary": ("The x86 SIMD instruction sets began with MMX; SSE introduced the 128-bit XMM "
                "vector registers and packed single-precision floating-point arithmetic, and AVX "
                "widened the registers further to 256-bit YMM, raising the parallel width of "
                "single-instruction-multiple-data generation by generation. They must be enabled "
                "through the CR0/CR4 control bits before use, and CPUID reports which of them the "
                "processor supports."),
    "children": ["ADDPS", "MOVAPS", "VADDPS", "SSE", "AVX", "XMM", "YMM"],
}


# ---------------------------------------------------------------------------
# Flat retrieval: treat each entity description as an independent chunk and
# recall it by lexical similarity (term-frequency cosine). This is a
# deterministic offline stand-in for vector search: no inherent structure, it
# only ever looks at the fragment itself.
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Coarse tokenizer: ASCII words (ADDPS, CR4, XMM) are kept whole, and CJK
    text is split per character so a Chinese knowledge base still works."""
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    tokens += re.findall(r"[\u4e00-\u9fff]", text)  # CJK run, one token per char
    return tokens


def _cosine(a: Dict[str, int], b: Dict[str, int]) -> float:
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    na = sum(v * v for v in a.values()) ** 0.5
    nb = sum(v * v for v in b.values()) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


class FlatRetriever:
    """Recalls independent chunks by lexical similarity (simulating flat vector search)."""

    def __init__(self, entities: Dict[str, Dict[str, str]]):
        self.docs = {name: e["desc"] for name, e in entities.items()}
        self.types = {name: e["type"] for name, e in entities.items()}
        self._vecs = {name: self._tf(text) for name, text in self.docs.items()}

    @staticmethod
    def _tf(text: str) -> Dict[str, int]:
        vec: Dict[str, int] = {}
        for tok in _tokenize(text):
            vec[tok] = vec.get(tok, 0) + 1
        return vec

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        qvec = self._tf(query)
        scored = [
            {"name": name, "type": self.types[name],
             "desc": self.docs[name], "score": _cosine(qvec, self._vecs[name])}
            for name in self.docs
        ]
        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored[:top_k]


# ---------------------------------------------------------------------------
# Structured retrieval: multi-hop traversal over the entity/relationship graph
# (the core capability of GraphRAG).
# ---------------------------------------------------------------------------

def build_graph(triples: List[Tuple[str, str, str]]) -> nx.DiGraph:
    g = nx.DiGraph()
    for name, meta in ENTITIES.items():
        g.add_node(name, **meta)
    for src, rel, dst in triples:
        g.add_edge(src, dst, rel=rel)
    return g


def multi_hop_paths(graph: nx.DiGraph, start: str, max_hops: int = 3) -> List[List[Tuple[str, str, str]]]:
    """Breadth-first walk along the relationship edges from `start`, returning
    every path of at most `max_hops` hops.

    Each path is a list of (source entity, relation, target entity) steps. This
    is exactly what flat retrieval cannot express -- "walking the relationship
    edges", which the book describes as knowledge graphs naturally supporting
    traversal along relationships, making multi-hop queries efficient and
    reliable.
    """
    if start not in graph:
        return []
    paths: List[List[Tuple[str, str, str]]] = []
    # Queue element: (current node, path taken to reach it)
    queue: deque = deque([(start, [])])
    while queue:
        node, path = queue.popleft()
        if len(path) >= max_hops:
            continue
        for nbr in graph.successors(node):
            step = (node, graph[node][nbr]["rel"], nbr)
            new_path = path + [step]
            paths.append(new_path)
            queue.append((nbr, new_path))
    return paths


def match_entity(graph: nx.DiGraph, query: str) -> Optional[str]:
    """Find the starting entity mentioned in the query (longest name match, deterministic)."""
    q = query.lower()
    hits = [name for name in graph.nodes if name.lower() in q]
    return max(hits, key=len) if hits else None


def format_path(path: List[Tuple[str, str, str]]) -> str:
    if not path:
        return ""
    parts = [path[0][0]]
    for src, rel, dst in path:
        parts.append(f" --{rel}--> {dst}")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Three demo queries, each highlighting a different shortcoming of flat retrieval.
# ---------------------------------------------------------------------------

def demo_multi_hop(flat: FlatRetriever, graph: nx.DiGraph, query: str, top_k: int) -> None:
    print(f"\n[Query 1 | Multi-hop relational reasoning] {query}")
    print("-- Flat retrieval (independent fragments by lexical similarity) --")
    for i, r in enumerate(flat.search(query, top_k), 1):
        print(f"  {i}. [{r['type']}] {r['name']}  (score={r['score']:.3f})")
    print("  X It can only recall lexically similar isolated fragments; it cannot *connect* "
          "ADDPS to a particular control bit. Without the relationship there is no way to tell "
          "which control bit is the answer for ADDPS.")

    print("-- Structured graph retrieval (multi-hop walk along relationship edges) --")
    start = match_entity(graph, query)
    paths = multi_hop_paths(graph, start, max_hops=3)
    # Show only paths ending at a control bit (the question asks about a control register bit)
    answers = [p for p in paths if graph.nodes[p[-1][2]]["type"] == "control-bit"]
    for p in answers:
        print(f"  {format_path(p)}")
    enable = [p for p in answers if p[-1][1] == "requires_enabling"]
    if enable:
        target = enable[0][-1][2]
        print(f"  = Answer: {target} (reachable from {start} in {len(enable[0])} hops)")
        print(f"    {graph.nodes[target]['desc']}")


def demo_compare(flat: FlatRetriever, graph: nx.DiGraph, query: str, top_k: int) -> None:
    print(f"\n[Query 2 | Synthesis across nodes] {query}")
    print("-- Flat retrieval --")
    for i, r in enumerate(flat.search(query, top_k), 1):
        print(f"  {i}. [{r['type']}] {r['name']}  (score={r['score']:.3f})")
    print("  X The facts about SSE's and AVX's registers live in separate fragments. Flat "
          "retrieval recalls them individually but never lines up 'which one uses which "
          "register' into a comparison.")

    print("-- Structured graph retrieval (follow the 'uses_register' edge on both sides) --")
    for ext in ("SSE", "AVX"):
        regs = [dst for _, dst, d in graph.out_edges(ext, data=True) if d["rel"] == "uses_register"]
        for reg in regs:
            print(f"  {ext} --uses_register--> {reg}: {graph.nodes[reg]['desc']}")
    print("  = Walking the same relationship edge from two entities synthesizes the comparison "
          "directly: SSE = 128-bit XMM, AVX = 256-bit YMM.")


def demo_hierarchical(flat: FlatRetriever, query: str, top_k: int) -> None:
    print(f"\n[Query 3 | Multi-level navigation (RAPTOR tree)] {query}")
    print("-- Flat retrieval --")
    for i, r in enumerate(flat.search(query, top_k), 1):
        print(f"  {i}. [{r['type']}] {r['name']}  (score={r['score']:.3f})")
    print("  X What comes back are scattered detail fragments, too fine-grained to answer an "
          "'overview' question that requires synthesis across fragments.")

    print("-- Structured tree retrieval (return the higher-level summary node) --")
    print(f"  [parent summary] {TREE_SUMMARY['id']}")
    print(f"  {TREE_SUMMARY['summary']}")
    print(f"  = Start from the broad summary, then drill down to leaves such as "
          f"{', '.join(TREE_SUMMARY['children'][:4])} when the detail is needed.")


def run_demo(top_k: int = 3, custom_query: Optional[str] = None,
             output: Optional[str] = None) -> Dict:
    """Run the offline comparison demo; returns structured results (for --output)."""
    flat = FlatRetriever(ENTITIES)
    graph = build_graph(TRIPLES)

    print("=" * 68)
    print("Structured index vs flat retrieval - offline comparison (no API key needed)")
    print(f"Knowledge base: Intel x86 SIMD instruction set  |  {graph.number_of_nodes()} entities, "
          f"{graph.number_of_edges()} relationships, 1 hierarchical tree")
    print("=" * 68)

    if custom_query:
        # Custom query: show both the flat and the graph perspective
        print(f"\n[Custom query] {custom_query}")
        print("-- Flat retrieval --")
        flat_hits = flat.search(custom_query, top_k)
        for i, r in enumerate(flat_hits, 1):
            print(f"  {i}. [{r['type']}] {r['name']}  (score={r['score']:.3f})")
        print("-- Structured graph retrieval (multi-hop walk from the entity found in the query) --")
        start = match_entity(graph, custom_query)
        if start is None:
            print("  (no known entity recognized in the query, so the graph cannot be walked)")
            paths = []
        else:
            paths = multi_hop_paths(graph, start, max_hops=3)
            for p in paths:
                print(f"  {format_path(p)}")
        result = {"query": custom_query,
                  "flat": [{"name": r["name"], "score": r["score"]} for r in flat_hits],
                  "graph_start": start,
                  "graph_paths": [format_path(p) for p in paths]}
    else:
        q1 = "Before running the ADDPS instruction, which control register bit must the operating system set to 1?"
        q2 = "What is the difference between the vector registers used by SSE and AVX?"
        q3 = "Give me an overview of the x86 SIMD instruction sets"
        demo_multi_hop(flat, graph, q1, top_k)
        demo_compare(flat, graph, q2, top_k)
        demo_hierarchical(flat, q3, top_k)
        start1 = match_entity(graph, q1)
        result = {
            "queries": [q1, q2, q3],
            "multi_hop": {
                "query": q1,
                "start": start1,
                "paths": [format_path(p) for p in multi_hop_paths(graph, start1, 3)
                          if graph.nodes[p[-1][2]]["type"] == "control-bit"],
            },
        }

    print("\n" + "=" * 68)
    print("Conclusion: flat retrieval is good at 'finding the fragment that contains some "
          "information', but as soon as a query needs relational reasoning across fragments or "
          "multi-level synthesis, it has to rely on a structured index (graph / hierarchical "
          "tree). This is the central point of experiment 3-8 in the book.")
    print("=" * 68)

    if output:
        with open(output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nResults written to {output}")
    return result


if __name__ == "__main__":
    run_demo()

"""Reranker module using BGE-Reranker-v2 model."""

import torch
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
import time
import numpy as np
from pathlib import Path
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

# FlagEmbedding is the preferred backend, but it is not always usable: it may be
# missing entirely (ModuleNotFoundError), and version 1.2.11 raises
# "NameError: name 'Optional' is not defined" against transformers >= 5. Fall
# back to running the same cross-encoder weights through transformers, which is
# the approach evaluate.py in this folder already uses.
try:
    from FlagEmbedding import FlagReranker
    FLAG_EMBEDDING_ERROR = None
except Exception as _flag_import_error:  # ModuleNotFoundError, NameError, ...
    FlagReranker = None
    FLAG_EMBEDDING_ERROR = _flag_import_error


class TransformersReranker:
    """Stand-in for FlagReranker built on transformers.

    Exposes the same compute_score(pairs, max_length=...) contract so Reranker
    does not care which backend is active.
    """

    def __init__(self, model_name: str, use_fp16: bool = False, device: str = "cpu"):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        if use_fp16 and device != "cpu":
            self.model = self.model.half()
        self.model = self.model.to(device).eval()

    def compute_score(self, pairs, max_length: int = 512, batch_size: int = 16):
        """Score [query, document] pairs; returns one float per pair."""
        if not pairs:
            return []
        scores = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            enc = self.tokenizer([p[0] for p in batch], [p[1] for p in batch],
                                 padding=True, truncation=True,
                                 max_length=max_length, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits.float()
            # transformers 5.x with some BERT weights can emit NaN in fp32;
            # recompute on CPU in float64 when that happens (same guard as evaluate.py)
            if torch.isnan(logits).any():
                logger.warning("Reranker produced NaN; recomputing on CPU in float64")
                enc_cpu = {k: v.to("cpu") for k, v in enc.items()}
                model64 = self.model.to("cpu").double()
                with torch.no_grad():
                    logits = model64(**enc_cpu).logits
                self.model = self.model.to(self.device).float()
            batch_scores = self._to_scores(logits)
            if len(batch_scores) != len(batch):
                raise RuntimeError(
                    f"Reranker produced {len(batch_scores)} scores for {len(batch)} "
                    f"pairs (logits shape {tuple(logits.shape)}); scores would be "
                    f"misaligned with documents."
                )
            scores.extend(batch_scores)
        return scores

    @staticmethod
    def _to_scores(logits) -> List[float]:
        """Reduce a logits tensor to exactly one relevance score per pair.

        Cross-encoder rerankers emit a single logit per pair (shape [N] or
        [N, 1]). A model with a 2-class head emits [N, 2]; take the positive
        class. Flattening blindly would return 2N scores and silently misalign
        every score with the wrong document.
        """
        if logits.ndim == 1:
            return [float(x) for x in logits.tolist()]
        if logits.shape[-1] == 1:
            return [float(x) for x in logits.squeeze(-1).tolist()]
        return [float(row[-1]) for row in logits.tolist()]  # positive class


@dataclass
class RerankResult:
    """Result from reranking."""
    doc_id: str
    rerank_score: float
    original_dense_score: Optional[float] = None
    original_sparse_score: Optional[float] = None
    original_dense_rank: Optional[int] = None
    original_sparse_rank: Optional[int] = None
    text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    debug_info: Optional[Dict[str, Any]] = None

class Reranker:
    """Reranker using BGE-Reranker-v2 model."""
    
    def _ensure_model_downloaded(self, model_name: str):
        """Check if model is cached and download if needed with progress.
        
        Args:
            model_name: HuggingFace model name
        """
        # Check cache directory
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        model_id = model_name.replace("/", "--")
        model_cache_path = cache_dir / f"models--{model_id}"
        
        if model_cache_path.exists() and any(model_cache_path.iterdir()):
            logger.info(f"Model already cached at {model_cache_path}")
            return
        
        logger.info(f"Model not found in cache. Downloading {model_name}...")
        logger.info("This is a one-time download. The model will be cached for future use.")
        
        try:
            # Use huggingface_hub to download with progress
            class DownloadProgressBar:
                def __init__(self):
                    self.pbar = None
                    self.total_size = 0
                    self.downloaded = 0
                
                def __call__(self, chunk_size: int):
                    if self.pbar is None:
                        return
                    self.downloaded += chunk_size
                    self.pbar.update(chunk_size)
                    
            # Download the model with progress tracking
            logger.info("Downloading model files...")
            # Fetch only what is needed to run the model. Without allow_patterns
            # this pulls the whole repo, including README images and any extra
            # checkpoint formats -- hundreds of MB of waste on a 2.3GB model.
            # resume_download is deprecated and ignored; downloads always resume.
            snapshot_download(
                repo_id=model_name,
                cache_dir=cache_dir,
                local_files_only=False,
                allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors"],
                ignore_patterns=["*.h5", "*.ot", "*.msgpack", "*.onnx", "assets/*", "*.png"],
            )
            logger.info("Model download completed!")
            
        except Exception as e:
            logger.warning(f"Could not pre-download model: {e}")
            logger.info("Model will be downloaded automatically during initialization...")
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", 
                 device: str = None, 
                 use_fp16: bool = True,
                 max_length: int = 512):
        """Initialize the reranker.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use (mps for Mac, cuda for GPU, cpu)
            use_fp16: Use half precision for faster inference
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        
        # Auto-detect device if not specified
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        
        self.device = device
        self.use_fp16 = use_fp16 and device != "cpu"
        self.max_length = max_length
        
        logger.info(f"Initializing reranker with model: {model_name}")
        logger.info(f"Device: {device}, FP16: {self.use_fp16}")
        
        # Check if model needs to be downloaded
        self._ensure_model_downloaded(model_name)
        
        # Initialize the model
        logger.info("Loading reranker model into memory...")
        start_time = time.time()
        if FlagReranker is not None:
            self.backend = "FlagEmbedding"
            self.model = FlagReranker(
                model_name,
                use_fp16=self.use_fp16,
                device=device
            )
        else:
            self.backend = "transformers"
            logger.warning(
                f"FlagEmbedding unavailable ({FLAG_EMBEDDING_ERROR}); reranking via "
                f"transformers instead. Fix with: pip install -U FlagEmbedding "
                f"(or pin transformers<5)."
            )
            self.model = TransformersReranker(
                model_name,
                use_fp16=self.use_fp16,
                device=device
            )
        elapsed = time.time() - start_time
        logger.info(f"Reranker initialized successfully in {elapsed:.2f}s "
                    f"(backend: {self.backend})")
    
    def rerank(self, 
               query: str, 
               documents: List[Dict[str, Any]], 
               top_k: int = 10,
               return_scores: bool = True) -> List[RerankResult]:
        """Rerank documents for a query.
        
        Args:
            query: The search query
            documents: List of documents with text and metadata
            top_k: Number of top results to return
            return_scores: Whether to return all scores for educational purposes
        
        Returns:
            List of reranked results
        """
        if not documents:
            return []
        
        start_time = time.time()
        logger.info(f"Reranking {len(documents)} documents for query: '{query[:50]}...'")
        
        # Prepare texts for reranking
        texts = []
        doc_info = []
        
        for doc in documents:
            text = doc.get("text", "")
            if not text:
                continue
                
            texts.append(text)
            doc_info.append({
                "doc_id": doc.get("doc_id"),
                "original_dense_score": doc.get("dense_score"),
                "original_sparse_score": doc.get("sparse_score"),
                "original_dense_rank": doc.get("dense_rank"),
                "original_sparse_rank": doc.get("sparse_rank"),
                "text": text,
                "metadata": doc.get("metadata", {})
            })
        
        if not texts:
            logger.warning("No valid texts to rerank")
            return []
        
        # Create query-document pairs
        pairs = [[query, text] for text in texts]
        
        # Get reranking scores
        try:
            scores = self.model.compute_score(pairs, max_length=self.max_length)
            
            # Convert to numpy array if needed
            if not isinstance(scores, np.ndarray):
                scores = np.array(scores)
            
            # Ensure scores is 1D. FlagReranker.compute_score returns a bare
            # float when exactly one pair is scored, which becomes a 0-d array
            # here — atleast_1d keeps the single-candidate case iterable.
            scores = np.atleast_1d(np.asarray(scores).squeeze())

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return []
        
        # Create results with scores
        results = []
        for i, score in enumerate(scores):
            info = doc_info[i]
            
            result = RerankResult(
                doc_id=info["doc_id"],
                rerank_score=float(score),
                original_dense_score=info["original_dense_score"],
                original_sparse_score=info["original_sparse_score"],
                original_dense_rank=info["original_dense_rank"],
                original_sparse_rank=info["original_sparse_rank"],
                text=info["text"] if return_scores else None,
                metadata=info["metadata"],
                debug_info={
                    "rerank_model": self.model_name,
                    "max_length": self.max_length,
                    "device": self.device
                }
            )
            results.append(result)
        
        # Sort by rerank score (descending)
        results.sort(key=lambda x: x.rerank_score, reverse=True)
        
        # Add final ranks
        for i, result in enumerate(results):
            if result.debug_info:
                result.debug_info["final_rank"] = i + 1
        
        elapsed_time = time.time() - start_time
        logger.info(f"Reranking completed in {elapsed_time:.2f}s")
        
        # Log score distribution for educational purposes
        if return_scores and results:
            scores_array = [r.rerank_score for r in results]
            logger.info(f"Rerank score distribution: min={min(scores_array):.3f}, "
                       f"max={max(scores_array):.3f}, mean={np.mean(scores_array):.3f}")
            
            # Log rank changes for top results
            for i, result in enumerate(results[:5]):
                changes = []
                if result.original_dense_rank:
                    dense_change = result.original_dense_rank - (i + 1)
                    changes.append(f"dense: {result.original_dense_rank}→{i+1} ({dense_change:+d})")
                if result.original_sparse_rank:
                    sparse_change = result.original_sparse_rank - (i + 1)
                    changes.append(f"sparse: {result.original_sparse_rank}→{i+1} ({sparse_change:+d})")
                
                if changes:
                    logger.debug(f"Doc {result.doc_id} rank changes: {', '.join(changes)}")
        
        # Return top_k results
        return results[:top_k]
    
    def batch_rerank(self, 
                     queries: List[str], 
                     documents_list: List[List[Dict[str, Any]]], 
                     top_k: int = 10,
                     batch_size: int = 32) -> List[List[RerankResult]]:
        """Rerank multiple queries in batch.
        
        Args:
            queries: List of queries
            documents_list: List of document lists (one per query)
            top_k: Number of top results per query
            batch_size: Batch size for processing
        
        Returns:
            List of reranked results for each query
        """
        all_results = []
        
        for query, documents in zip(queries, documents_list):
            results = self.rerank(query, documents, top_k)
            all_results.append(results)
        
        return all_results

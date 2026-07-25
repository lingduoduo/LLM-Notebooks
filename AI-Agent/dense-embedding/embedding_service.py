"""Embedding service using BGE-M3 model."""

import time
import numpy as np
from typing import List, Dict, Optional
from logger import VectorSearchLogger, log_execution_time
import logging

# FlagEmbedding is the preferred backend (it also produces sparse and ColBERT
# vectors), but it is not always importable: FlagEmbedding 1.2.11 raises
# "NameError: name 'Optional' is not defined" against transformers >= 5, and it
# may simply not be installed. Fall back to loading the same BGE-M3 weights
# through transformers, which covers the dense vectors this service needs.
try:
    from FlagEmbedding import BGEM3FlagModel
    FLAG_EMBEDDING_ERROR = None
except Exception as _flag_import_error:  # ImportError, NameError, ...
    BGEM3FlagModel = None
    FLAG_EMBEDDING_ERROR = _flag_import_error


class TransformersBGEM3:
    """Dense-only stand-in for BGEM3FlagModel, built on plain transformers.

    Exposes the same encode() signature so EmbeddingService does not care which
    backend is active. BGE-M3 pools the [CLS] token and L2-normalizes, which is
    what FlagEmbedding returns in 'dense_vecs'.
    """

    def __init__(self, model_name: str, use_fp16: bool = False, max_seq_length: int = 512):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.max_seq_length = max_seq_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        if use_fp16 and torch.cuda.is_available():
            self.model = self.model.half()
        self.model.eval()

    def encode(self, texts, return_dense: bool = True, return_sparse: bool = False,
               return_colbert_vecs: bool = False, batch_size: int = 16, **kwargs) -> Dict:
        if return_sparse or return_colbert_vecs:
            raise NotImplementedError(
                "Sparse and ColBERT vectors need the FlagEmbedding backend, which failed "
                f"to import ({FLAG_EMBEDDING_ERROR}). Only dense vectors are available. "
                "Fix with: pip install -U FlagEmbedding  (or pin transformers<5)."
            )
        torch = self.torch
        if isinstance(texts, str):
            texts = [texts]
        vectors = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start:start + batch_size])
            enc = self.tokenizer(batch, padding=True, truncation=True,
                                 max_length=self.max_seq_length, return_tensors="pt")
            with torch.no_grad():
                out = self.model(**enc)
            pooled = out.last_hidden_state[:, 0]  # BGE-M3 uses CLS pooling
            pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
            vectors.append(pooled.cpu().numpy().astype("float32"))
        return {"dense_vecs": np.vstack(vectors)}


class EmbeddingService:
    """Service for generating embeddings using BGE-M3 model."""
    
    def __init__(self, model_name: str = "BAAI/bge-m3", use_fp16: bool = True, 
                 max_seq_length: int = 512, logger: Optional[VectorSearchLogger] = None):
        """
        Initialize the embedding service with BGE-M3 model.
        
        Args:
            model_name: Name of the BGE-M3 model
            use_fp16: Whether to use FP16 for inference
            max_seq_length: Maximum sequence length
            logger: Logger instance for educational output
        """
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.max_seq_length = max_seq_length
        self.logger = logger
        self.std_logger = logging.getLogger("vector_search")
        
        # Initialize the model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the BGE-M3 model."""
        start_time = time.time()
        
        if self.logger:
            self.logger.logger.info(f"🚀 Initializing BGE-M3 model: {self.model_name}")
            self.logger.logger.debug(f"  - Using FP16: {self.use_fp16}")
            self.logger.logger.debug(f"  - Max sequence length: {self.max_seq_length}")
        
        try:
            if BGEM3FlagModel is not None:
                self.backend = "FlagEmbedding"
                self.model = BGEM3FlagModel(
                    self.model_name,
                    use_fp16=self.use_fp16
                )
            else:
                self.backend = "transformers"
                msg = (f"FlagEmbedding unavailable ({FLAG_EMBEDDING_ERROR}); "
                       f"falling back to transformers for dense vectors. "
                       f"Sparse/ColBERT vectors are disabled. "
                       f"Fix with: pip install -U FlagEmbedding (or pin transformers<5).")
                self.std_logger.warning(msg)
                if self.logger:
                    self.logger.logger.warning(f"⚠️  {msg}")
                self.model = TransformersBGEM3(
                    self.model_name,
                    use_fp16=self.use_fp16,
                    max_seq_length=self.max_seq_length,
                )


            # Get embedding dimension by encoding a test sentence
            test_embedding = self.model.encode(["test"])
            if isinstance(test_embedding, dict):
                self.embedding_dim = test_embedding['dense_vecs'].shape[1]
            else:
                self.embedding_dim = test_embedding.shape[1]
            
            load_time = time.time() - start_time
            
            if self.logger:
                self.logger.logger.info(f"✅ Model loaded successfully in {load_time:.2f} seconds")
                self.logger.logger.debug(f"  - Embedding dimension: {self.embedding_dim}")
                self.logger.logger.debug(f"  - Model supports: dense, sparse, and multi-vector retrieval")
                
        except Exception as e:
            if self.logger:
                self.logger.logger.error(f"Failed to load model: {e}")
            raise
    
    @log_execution_time()
    def encode_text(self, text: str, return_sparse: bool = False, 
                   return_colbert: bool = False) -> Dict[str, np.ndarray]:
        """
        Encode a single text into embeddings.
        
        Args:
            text: Input text to encode
            return_sparse: Whether to return sparse embeddings
            return_colbert: Whether to return ColBERT embeddings
        
        Returns:
            Dictionary containing different types of embeddings
        """
        start_time = time.time()
        
        if self.logger:
            self.logger.logger.debug(f"📝 Encoding text (length: {len(text)} chars)")
            self.logger.logger.debug(f"  Text preview: {text[:100]}..." if len(text) > 100 else f"  Text: {text}")
        
        # Encode the text
        embeddings = self.model.encode(
            [text],
            return_dense=True,
            return_sparse=return_sparse,
            return_colbert_vecs=return_colbert
        )
        
        # Extract dense embeddings
        dense_vec = embeddings['dense_vecs'][0]
        
        result = {
            'dense': dense_vec,
            'dimension': len(dense_vec)
        }
        
        # Add sparse embeddings if requested
        if return_sparse and 'lexical_weights' in embeddings:
            result['sparse'] = embeddings['lexical_weights'][0]
            if self.logger:
                num_tokens = len(result['sparse'])
                self.logger.logger.debug(f"  Sparse embedding: {num_tokens} non-zero tokens")
        
        # Add ColBERT embeddings if requested  
        if return_colbert and 'colbert_vecs' in embeddings:
            result['colbert'] = embeddings['colbert_vecs'][0]
            if self.logger:
                colbert_shape = result['colbert'].shape
                self.logger.logger.debug(f"  ColBERT embedding shape: {colbert_shape}")
        
        encoding_time = time.time() - start_time
        
        if self.logger:
            self.logger.logger.debug(f"✅ Encoding completed in {encoding_time:.4f} seconds")
            self.logger.log_embedding_vector(dense_vec, sample_size=10)
        
        return result
    
    @log_execution_time()
    def encode_batch(self, texts: List[str], return_sparse: bool = False, 
                    return_colbert: bool = False) -> Dict[str, np.ndarray]:
        """
        Encode multiple texts into embeddings.
        
        Args:
            texts: List of input texts to encode
            return_sparse: Whether to return sparse embeddings
            return_colbert: Whether to return ColBERT embeddings
        
        Returns:
            Dictionary containing different types of embeddings for all texts
        """
        start_time = time.time()
        
        if self.logger:
            self.logger.logger.info(f"📚 Batch encoding {len(texts)} texts")
            total_chars = sum(len(t) for t in texts)
            self.logger.logger.debug(f"  Total characters: {total_chars}")
            self.logger.logger.debug(f"  Average text length: {total_chars/len(texts):.1f} chars")
        
        # Encode all texts
        embeddings = self.model.encode(
            texts,
            return_dense=True,
            return_sparse=return_sparse,
            return_colbert_vecs=return_colbert
        )
        
        result = {
            'dense': embeddings['dense_vecs'],
            'dimension': embeddings['dense_vecs'].shape[1],
            'num_texts': len(texts)
        }
        
        # Add sparse embeddings if requested
        if return_sparse and 'lexical_weights' in embeddings:
            result['sparse'] = embeddings['lexical_weights']
        
        # Add ColBERT embeddings if requested
        if return_colbert and 'colbert_vecs' in embeddings:
            result['colbert'] = embeddings['colbert_vecs']
        
        encoding_time = time.time() - start_time
        
        if self.logger:
            self.logger.logger.info(f"✅ Batch encoding completed in {encoding_time:.4f} seconds")
            self.logger.logger.debug(f"  Average time per text: {encoding_time/len(texts):.4f} seconds")
        
        return result
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        return self.embedding_dim
    
    def compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray, 
                          metric: str = "cosine") -> float:
        """
        Compute similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            metric: Similarity metric ('cosine', 'euclidean', 'dot')
        
        Returns:
            Similarity score
        """
        if metric == "cosine":
            # Cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            similarity = dot_product / (norm1 * norm2)
            
        elif metric == "euclidean":
            # Euclidean distance (negative for similarity)
            similarity = -np.linalg.norm(vec1 - vec2)
            
        elif metric == "dot":
            # Dot product
            similarity = np.dot(vec1, vec2)
            
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if self.logger:
            self.logger.logger.debug(f"  Similarity ({metric}): {similarity:.6f}")
        
        return float(similarity)

"""
Deep NLP Feature Engineering
============================
Uses HuggingFace Sentence Transformers to generate embeddings for campaign text.

This allows the causal model to "understand" the quality and topic of a campaign
purely from its text, acting as a powerful confounder control.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

class ContentEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the embedder with a lightweight BERT model.
        
        Args:
            model_name: HuggingFace model identifier.
                        'all-MiniLM-L6-v2' is fast and good quality.
        """
        self.model_name = model_name
        self.model = None
        
    def load_model(self):
        """Lazy load the model only when needed."""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading NLP model: {self.model_name}...")
                self.model = SentenceTransformer(self.model_name)
                logger.info("NLP model loaded successfully.")
            except ImportError:
                logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
                raise
            except Exception as e:
                logger.error(f"Failed to load NLP model: {e}")
                raise

    def generate_embeddings(self, texts, limit=None):
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts (list/Series): Text content to embed.
            limit (int): Max number of dimensions to return (PCA reduction could be added here).
            
        Returns:
            np.ndarray: Matrix of embeddings.
        """
        self.load_model()
        
        # Ensure texts are strings
        texts = [str(t) if pd.notna(t) else "" for t in texts]
        
        logger.info(f"Generating embeddings for {len(texts)} items...")
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
        
        if limit and limit < embeddings.shape[1]:
            # Simple truncation (PCA would be better but requires training)
            # For simplicity in this pipeline, we'll just take the top N components
            # assuming the model's output is somewhat distributed.
            # A better approach involves PCA, but let's stick to raw embeddings for now
            # or use a smaller PCA if needed.
            # Actually, Causal Forest usually handles high dimensions well.
            pass
            
        return embeddings

    def process_dataframe(self, df, text_cols=['name', 'desc'], n_components=10):
        """
        Add NLP embedding features to dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe.
            text_cols (list): Columns to combine for embedding.
            n_components (int): Number of PCA components to keep (to avoid 384 dimensions).
            
        Returns:
            pd.DataFrame: Dataframe with new columns 'nlp_0', 'nlp_1', ...
        """
        from sklearn.decomposition import PCA
        
        # Combine text columns
        combined_text = df[text_cols].fillna('').astype(str).agg(' '.join, axis=1)
        
        # Generate raw embeddings (384 dim for MiniLM)
        raw_embeddings = self.generate_embeddings(combined_text)
        
        print(f"DEBUG: df_len={len(df)}, n_components={n_components}, embedding_shape={raw_embeddings.shape}")

        # Reduce dimensionality with PCA
        if len(df) > n_components:
            pca = PCA(n_components=n_components)
            reduced_embeddings = pca.fit_transform(raw_embeddings)
            
            # Create new columns
            new_cols = []
            for i in range(n_components):
                col_name = f'nlp_dim_{i}'
                df[col_name] = reduced_embeddings[:, i]
                new_cols.append(col_name)
            
            return df, new_cols
        else:
            print("DEBUG: Not enough samples for PCA")
            return df, []

# Feature Integration
def add_nlp_features(df, text_cols=['name', 'desc'], n_components=15):
    """Wrapper function to be called from pipeline."""
    embedder = ContentEmbedder()
    df_enriched, new_cols = embedder.process_dataframe(df, text_cols=text_cols, n_components=n_components)
    return df_enriched, new_cols

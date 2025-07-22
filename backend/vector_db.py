import os
import faiss
import pickle
import pandas as pd
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
import requests
from dotenv import load_dotenv
import time
from functools import lru_cache
import threading

load_dotenv()

class VectorDB_FAISS:
    def __init__(self, index_path: str = "faiss_index", max_cache_size: int = 1000):
        self.index_path = index_path
        self.dimension = 384  # MiniLM-L6-v2 has 384 dims
        self.max_cache_size = max_cache_size
        
        # Lazy load model to reduce startup time
        self._model = None
        self._model_lock = threading.Lock()
        
        # Initialize empty structures
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        self.metadatas = []
        self.ids = []

        # Load existing index if available
        self._load_index()

    @property
    def model(self):
        """Lazy loading of sentence transformer model"""
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    print("🔄 Loading sentence transformer model...")
                    self._model = SentenceTransformer("all-MiniLM-L6-v2")
                    print("✅ Model loaded successfully")
        return self._model

    def _load_index(self):
        """Load existing FAISS index and metadata"""
        index_file = self.index_path + ".index"
        metadata_file = self.index_path + ".pkl"

        if os.path.exists(index_file) and os.path.exists(metadata_file):
            try:
                print("🔄 Loading existing FAISS index...")
                start_time = time.time()
                
                self.index = faiss.read_index(index_file)
                with open(metadata_file, "rb") as f:
                    data = pickle.load(f)
                    
                    if isinstance(data, tuple) and len(data) == 3:
                        self.documents, self.metadatas, self.ids = data
                    else:
                        self.documents = data.get('documents', [])
                        self.metadatas = data.get('metadatas', [])
                        self.ids = data.get('ids', [])
                
                load_time = time.time() - start_time
                print(f"✅ Loaded FAISS index in {load_time:.2f}s with {len(self.documents)} documents")
                print(f"📊 Index contains {self.index.ntotal} vectors")
                
                # Verify data consistency
                if len(self.documents) != self.index.ntotal:
                    print(f"⚠️ Warning: Document count ({len(self.documents)}) doesn't match index size ({self.index.ntotal})")
                        
            except Exception as e:
                print(f"❌ Error loading index metadata: {e}")
                self._reset_index()
        else:
            print("ℹ️ No existing FAISS index found. Starting fresh.")

    def _reset_index(self):
        """Reset index to empty state"""
        self.documents = []
        self.metadatas = []
        self.ids = []
        self.index = faiss.IndexFlatL2(self.dimension)

    def _save_index(self):
        """Save FAISS index and metadata"""
        try:
            start_time = time.time()
            faiss.write_index(self.index, self.index_path + ".index")
            with open(self.index_path + ".pkl", "wb") as f:
                pickle.dump((self.documents, self.metadatas, self.ids), f)
            save_time = time.time() - start_time
            print(f"✅ Saved FAISS index in {save_time:.2f}s")
        except Exception as e:
            print(f"❌ Error saving index: {e}")

    def add_documents(self, documents: List[str], metadatas: List[Dict], ids: List[str]):
        """Add documents to the vector database with batch processing"""
        try:
            # Filter out empty documents
            valid_docs = []
            valid_metas = []
            valid_ids = []
            
            for doc, meta, doc_id in zip(documents, metadatas, ids):
                if doc and doc.strip():
                    valid_docs.append(doc.strip())
                    valid_metas.append(meta)
                    valid_ids.append(doc_id)
            
            if not valid_docs:
                print("⚠️ No valid documents to add")
                return

            print(f"🔄 Encoding {len(valid_docs)} documents...")
            start_time = time.time()
            
            # Batch encode for better performance
            embeddings = self.model.encode(
                valid_docs, 
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=32  # Process in batches
            ).astype("float32")
            
            encoding_time = time.time() - start_time
            print(f"📊 Generated embeddings in {encoding_time:.2f}s, shape: {embeddings.shape}")
            
            # Add to index
            self.index.add(embeddings)
            self.documents.extend(valid_docs)
            self.metadatas.extend(valid_metas)
            self.ids.extend(valid_ids)

            # Save to disk
            self._save_index()
            print(f"✅ Added {len(valid_docs)} documents to FAISS index")
            
        except Exception as e:
            print(f"❌ Error adding documents: {e}")
            import traceback
            traceback.print_exc()

    @lru_cache(maxsize=100)
    def _encode_query_cached(self, query: str) -> np.ndarray:
        """Cache query encodings for repeated queries"""
        return self.model.encode([query], convert_to_numpy=True).astype("float32")

    def search(self, query: str, n_results: int = 5):
        """Search the vector database with optimizations"""
        if len(self.documents) == 0 or self.index.ntotal == 0:
            print("⚠️ No documents in index to search")
            return []

        try:
            print(f"🔍 Searching for: '{query[:50]}...'")
            start_time = time.time()
            
            # Use cached encoding if available
            try:
                embedding = self._encode_query_cached(query)
            except:
                # Fallback to regular encoding if caching fails
                embedding = self.model.encode([query], convert_to_numpy=True).astype("float32")
            
            # Ensure we don't ask for more results than available
            actual_n_results = min(n_results, self.index.ntotal)
            
            # Perform search
            D, I = self.index.search(embedding, actual_n_results)
            search_time = time.time() - start_time
            
            print(f"🎯 Search completed in {search_time:.3f}s, found {len(I[0])} results")

            results = []
            for i, idx in enumerate(I[0]):
                if idx >= 0 and idx < len(self.documents):  # Valid index
                    result = {
                        "document": self.documents[idx],
                        "metadata": self.metadatas[idx] if idx < len(self.metadatas) else {},
                        "distance": float(D[0][i])
                    }
                    results.append(result)
                else:
                    print(f"⚠️ Skipped invalid index {idx}")

            return results
            
        except Exception as e:
            print(f"❌ Error during search: {e}")
            import traceback
            traceback.print_exc()
            return []

    # Use connection pooling for API calls
    _session = None
    _session_lock = threading.Lock()

    @property
    def session(self):
        """Lazy initialization of requests session with connection pooling"""
        if self._session is None:
            with self._session_lock:
                if self._session is None:
                    self._session = requests.Session()
                    adapter = requests.adapters.HTTPAdapter(
                        pool_connections=5,
                        pool_maxsize=10,
                        max_retries=2
                    )
                    self._session.mount('https://', adapter)
        return self._session

    @lru_cache(maxsize=1)
    def _get_api_config(self):
        """Cached API configuration"""
        return {
            'api_key': os.getenv('OPENROUTER_API_KEY'),
            'api_url': os.getenv('OPENROUTER_API_URL', 'https://openrouter.ai/api/v1/chat/completions'),
            'model': os.getenv('DEEPSEEK_MODEL', 'deepseek/deepseek-chat'),
            'site_url': os.getenv('YOUR_SITE_URL', 'http://localhost:5000'),
            'site_name': os.getenv('YOUR_SITE_NAME', 'Flask RAG App')
        }

    def _call_deepseek_api(self, prompt: str, context: str = "") -> str:
        """Optimized DeepSeek API call with connection pooling and retries"""
        try:
            config = self._get_api_config()

            if not config['api_key']:
                return "Error: OpenRouter API key not configured. Please set OPENROUTER_API_KEY in your environment variables."

            headers = {
                'Authorization': f'Bearer {config["api_key"]}',
                'Content-Type': 'application/json',
                'HTTP-Referer': config['site_url'],
                'X-Title': config['site_name']
            }

            # Construct prompt with context
            if context:
                full_prompt = f"""Based on the following context, please answer the user's question concisely. If the context doesn't contain relevant information, please say so briefly.

Context:
{context[:2000]}  # Limit context length

Question: {prompt}

Answer:"""
            else:
                full_prompt = prompt

            payload = {
                'model': config['model'],
                'messages': [
                    {
                        'role': 'user',
                        'content': full_prompt
                    }
                ],
                'max_tokens': 500,  # Reduced for faster response
                'temperature': 0.3,  # Lower temperature for more focused responses
                'stream': False
            }

            print(f"🤖 Calling DeepSeek API...")
            start_time = time.time()
            
            # Use session with connection pooling
            response = self.session.post(
                config['api_url'], 
                headers=headers, 
                json=payload, 
                timeout=12  # Reduced timeout
            )

            api_time = time.time() - start_time
            print(f"⏱️ API call completed in {api_time:.2f}s")

            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    content = result['choices'][0]['message']['content']
                    print("✅ DeepSeek API response received")
                    return content
                else:
                    return "I couldn't generate a response. Please try rephrasing your question."
            else:
                print(f"❌ API Error: {response.status_code}")
                return "I'm experiencing technical difficulties. Please try again later."

        except requests.exceptions.Timeout:
            print("❌ API request timed out")
            return "The response is taking too long. Please try a simpler question."
        except Exception as e:
            print(f"❌ API call error: {str(e)}")
            return "I encountered an error. Please try again."

    def rag_search_and_answer(self, query: str, n_results: int = 3, distance_threshold: float = 1.0) -> Dict:
        """Optimized RAG search and answer with better performance"""
        try:
            print(f"🔍 RAG Search for: '{query[:50]}...'")
            start_time = time.time()
            
            # Step 1: Search vector database
            search_results = self.search(query, n_results)
            
            # Step 2: Filter relevant results
            relevant_results = [r for r in search_results if r['distance'] < distance_threshold]
            
            if relevant_results:
                print(f"📚 Found {len(relevant_results)} relevant documents")
                
                # Step 3: Prepare concise context
                context_parts = []
                for i, result in enumerate(relevant_results[:2], 1):  # Limit to top 2 results
                    doc = result['document']
                    if 'Answer:' in doc:
                        parts = doc.split('Answer:')
                        if len(parts) == 2:
                            question = parts[0].replace('Question:', '').strip()
                            answer = parts[1].strip()
                            context_parts.append(f"Q: {question}\nA: {answer}")
                    else:
                        context_parts.append(doc[:200])  # Truncate long documents
                
                context = "\n\n".join(context_parts)
                
                # Step 4: Use DeepSeek with context
                answer = self._call_deepseek_api(query, context)
                
                total_time = time.time() - start_time
                print(f"⏱️ RAG completed in {total_time:.2f}s")
                
                return {
                    'answer': answer,
                    'method': 'rag_with_context',
                    'sources': relevant_results,
                    'context_used': True,
                    'num_sources': len(relevant_results)
                }
            else:
                print("🌐 No relevant context found, using direct API")
                answer = self._call_deepseek_api(query)
                
                return {
                    'answer': answer,
                    'method': 'api_only',
                    'sources': [],
                    'context_used': False,
                    'num_sources': 0
                }
                
        except Exception as e:
            print(f"❌ RAG error: {e}")
            return {
                'answer': "I encountered an error processing your request. Please try again.",
                'method': 'error',
                'sources': [],
                'context_used': False,
                'num_sources': 0
            }

    def hybrid_search(self, query: str, mode: str = "auto", n_results: int = 3, distance_threshold: float = 1.0) -> Dict:
        """Optimized hybrid search with multiple modes"""
        start_time = time.time()
        
        if mode == "vector_only":
            search_results = self.search(query, n_results)
            relevant_results = [r for r in search_results if r['distance'] < distance_threshold]
            
            if relevant_results:
                context = "Based on your data:\n\n"
                for i, result in enumerate(relevant_results[:3], 1):  # Limit to top 3
                    doc = result['document']
                    if 'Answer:' in doc:
                        parts = doc.split('Answer:')
                        if len(parts) == 2:
                            answer = parts[1].strip()
                            context += f"{i}. {answer}\n\n"
                    else:
                        context += f"{i}. {doc[:150]}...\n\n"  # Truncate
                
                return {
                    'answer': context,
                    'method': 'vector_only',
                    'sources': relevant_results,
                    'context_used': True,
                    'num_sources': len(relevant_results)
                }
            else:
                return {
                    'answer': "No relevant information found in the database.",
                    'method': 'vector_only',
                    'sources': [],
                    'context_used': False,
                    'num_sources': 0
                }
        
        elif mode == "deepseek_only":
            answer = self._call_deepseek_api(query)
            return {
                'answer': answer,
                'method': 'deepseek_only',
                'sources': [],
                'context_used': False,
                'num_sources': 0
            }
        
        elif mode == "rag":
            return self.rag_search_and_answer(query, n_results, distance_threshold)
        
        else:  # mode == "auto"
            # Quick relevance check
            search_results = self.search(query, min(n_results, 2))  # Limit initial search
            relevant_results = [r for r in search_results if r['distance'] < distance_threshold]
            
            if relevant_results:
                return self.rag_search_and_answer(query, n_results, distance_threshold)
            else:
                answer = self._call_deepseek_api(query)
                return {
                    'answer': answer,
                    'method': 'deepseek_fallback',
                    'sources': [],
                    'context_used': False,
                    'num_sources': 0
                }

    def index_csv(self, csv_path: str):
        """Optimized CSV indexing with batch processing"""
        if not os.path.exists(csv_path):
            return False, f"CSV path '{csv_path}' does not exist."

        try:
            print(f"📁 Reading CSV from: {csv_path}")
            start_time = time.time()
            
            # Read CSV with optimization
            df = pd.read_csv(csv_path, encoding='utf-8')
            print(f"📊 CSV loaded in {time.time() - start_time:.2f}s, shape: {df.shape}")

            # Find required columns (case-insensitive)
            df_columns_lower = [col.lower().strip() for col in df.columns]
            
            question_col = None
            answer_col = None
            
            # More flexible column matching
            for col in df.columns:
                col_lower = col.lower().strip()
                if col_lower in ['question', 'q', 'questions', 'ques']:
                    question_col = col
                elif col_lower in ['answer', 'a', 'response', 'answers', 'ans']:
                    answer_col = col
            
            if not question_col or not answer_col:
                available_cols = ', '.join(df.columns)
                return False, f"CSV must contain 'Question' and 'Answer' columns (case-insensitive). Available columns: {available_cols}"

            print(f"📋 Using columns - Question: '{question_col}', Answer: '{answer_col}'")

            # Process data in batches for better performance
            batch_size = 100
            total_valid = 0
            
            documents = []
            metadatas = []
            
            for batch_start in range(0, len(df), batch_size):
                batch_end = min(batch_start + batch_size, len(df))
                batch_df = df.iloc[batch_start:batch_end]
                
                batch_docs = []
                batch_metas = []
                
                for _, row in batch_df.iterrows():
                    question = str(row[question_col]).strip() if pd.notna(row[question_col]) else ""
                    answer = str(row[answer_col]).strip() if pd.notna(row[answer_col]) else ""
                    
                    # Skip empty or invalid entries
                    if (question and answer and 
                        question.lower() not in ['nan', 'none', ''] and 
                        answer.lower() not in ['nan', 'none', '']):
                        
                        # Create comprehensive document for better search
                        doc = f"Question: {question}\nAnswer: {answer}"
                        batch_docs.append(doc)
                        
                        # Create metadata
                        meta = {
                            'question': question,
                            'answer': answer,
                            'source': csv_path,
                            'row_index': total_valid
                        }
                        
                        # Add additional columns as metadata (optional)
                        for col in df.columns:
                            if col not in [question_col, answer_col]:
                                try:
                                    val = str(row[col]) if pd.notna(row[col]) else ""
                                    if val and val.lower() not in ['nan', 'none']:
                                        meta[col.lower().replace(' ', '_')] = val[:100]  # Limit length
                                except:
                                    pass  # Skip problematic columns
                        
                        batch_metas.append(meta)
                        total_valid += 1
                
                documents.extend(batch_docs)
                metadatas.extend(batch_metas)
                
                if batch_docs:
                    print(f"📦 Processed batch {batch_start//batch_size + 1}: {len(batch_docs)} valid entries")

            if not documents:
                return False, "No valid question-answer pairs found in CSV"

            # Generate IDs
            ids = [f"doc_{i:06d}" for i in range(len(documents))]

            print(f"📝 Prepared {len(documents)} documents for indexing")
            
            # Add to vector database
            self.add_documents(documents, metadatas, ids)
            
            csv_time = time.time() - start_time
            print(f"⏱️ CSV indexing completed in {csv_time:.2f}s")
            
            return True, f"Successfully indexed {len(documents)} Q&A pairs from {csv_path} in {csv_time:.2f}s"

        except Exception as e:
            print(f"❌ Error indexing CSV: {e}")
            import traceback
            traceback.print_exc()
            return False, f"CSV indexing failed: {str(e)}"

    def get_collection_info(self):
        """Get information about the current collection"""
        return {
            "count": len(self.documents),
            "index_size": self.index.ntotal,
            "dimension": self.dimension,
            "index_type": "FAISS L2",
            "model": "all-MiniLM-L6-v2",
            "cache_size": self.max_cache_size
        }

    def clear_index(self):
        """Clear the entire index and reset"""
        try:
            print("🗑️ Clearing FAISS index...")
            self._reset_index()
            self._save_index()
            print("✅ FAISS index cleared successfully")
            return True
        except Exception as e:
            print(f"❌ Error clearing index: {e}")
            return False

    def get_document_by_id(self, doc_id: str):
        """Get a specific document by ID"""
        try:
            idx = self.ids.index(doc_id)
            return {
                "document": self.documents[idx],
                "metadata": self.metadatas[idx],
                "id": doc_id,
                "index": idx
            }
        except ValueError:
            return None

    def delete_document(self, doc_id: str):
        """Delete a document by ID (requires rebuilding index)"""
        try:
            idx = self.ids.index(doc_id)
            
            # Remove from lists
            self.documents.pop(idx)
            self.metadatas.pop(idx)
            self.ids.pop(idx)
            
            # Rebuild index (FAISS doesn't support individual deletions)
            print("🔄 Rebuilding index after deletion...")
            embeddings = self.model.encode(
                self.documents, 
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=32
            ).astype("float32")
            
            # Create new index
            self.index = faiss.IndexFlatL2(self.dimension)
            if len(embeddings) > 0:
                self.index.add(embeddings)
            
            self._save_index()
            print(f"✅ Document {doc_id} deleted successfully")
            return True
            
        except ValueError:
            print(f"❌ Document {doc_id} not found")
            return False
        except Exception as e:
            print(f"❌ Error deleting document: {e}")
            return False

    def update_document(self, doc_id: str, new_document: str, new_metadata: Dict = None):
        """Update a document by ID (requires rebuilding index)"""
        try:
            idx = self.ids.index(doc_id)
            
            # Update document and metadata
            self.documents[idx] = new_document.strip()
            if new_metadata:
                self.metadatas[idx].update(new_metadata)
            
            # Rebuild index
            print("🔄 Rebuilding index after update...")
            embeddings = self.model.encode(
                self.documents, 
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=32
            ).astype("float32")
            
            # Create new index
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(embeddings)
            
            self._save_index()
            print(f"✅ Document {doc_id} updated successfully")
            return True
            
        except ValueError:
            print(f"❌ Document {doc_id} not found")
            return False
        except Exception as e:
            print(f"❌ Error updating document: {e}")
            return False

    def get_similar_documents(self, doc_id: str, n_results: int = 5):
        """Find documents similar to a given document"""
        try:
            doc = self.get_document_by_id(doc_id)
            if not doc:
                return []
            
            # Use the document text as query
            results = self.search(doc['document'], n_results + 1)  # +1 to exclude self
            
            # Filter out the original document
            similar_docs = [r for r in results if r['metadata'].get('row_index') != doc['metadata'].get('row_index')]
            
            return similar_docs[:n_results]
            
        except Exception as e:
            print(f"❌ Error finding similar documents: {e}")
            return []

    def export_to_csv(self, output_path: str):
        """Export current database to CSV"""
        try:
            if not self.documents:
                return False, "No documents to export"
            
            export_data = []
            for i, (doc, meta) in enumerate(zip(self.documents, self.metadatas)):
                row = {
                    'id': self.ids[i],
                    'question': meta.get('question', ''),
                    'answer': meta.get('answer', ''),
                    'document': doc,
                    'source': meta.get('source', ''),
                    **{k: v for k, v in meta.items() if k not in ['question', 'answer', 'source']}
                }
                export_data.append(row)
            
            df = pd.DataFrame(export_data)
            df.to_csv(output_path, index=False)
            
            print(f"✅ Exported {len(export_data)} documents to {output_path}")
            return True, f"Successfully exported {len(export_data)} documents"
            
        except Exception as e:
            print(f"❌ Error exporting to CSV: {e}")
            return False, str(e)

    def optimize_index(self):
        """Optimize the FAISS index for better performance"""
        try:
            if self.index.ntotal == 0:
                print("⚠️ No data to optimize")
                return False
            
            print("🔄 Optimizing FAISS index...")
            start_time = time.time()
            
            # For larger datasets, consider using IndexIVFFlat for better performance
            if self.index.ntotal > 10000:
                print("📊 Large dataset detected, using IVF index...")
                
                # Create quantizer
                quantizer = faiss.IndexFlatL2(self.dimension)
                
                # Create IVF index
                nlist = min(100, int(np.sqrt(self.index.ntotal)))  # Number of clusters
                new_index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
                
                # Get all vectors from current index
                all_vectors = np.zeros((self.index.ntotal, self.dimension), dtype=np.float32)
                for i in range(self.index.ntotal):
                    all_vectors[i] = self.index.reconstruct(i)
                
                # Train and add to new index
                new_index.train(all_vectors)
                new_index.add(all_vectors)
                
                # Replace old index
                self.index = new_index
                self._save_index()
                
                opt_time = time.time() - start_time
                print(f"✅ Index optimized to IVF in {opt_time:.2f}s")
            else:
                print("📊 Dataset size is optimal for L2 index")
            
            return True
            
        except Exception as e:
            print(f"❌ Error optimizing index: {e}")
            return False

    def get_statistics(self):
        """Get detailed statistics about the database"""
        try:
            if not self.documents:
                return {}
            
            # Document length statistics
            doc_lengths = [len(doc) for doc in self.documents]
            
            # Metadata statistics
            question_lengths = [len(meta.get('question', '')) for meta in self.metadatas]
            answer_lengths = [len(meta.get('answer', '')) for meta in self.metadatas]
            
            stats = {
                'total_documents': len(self.documents),
                'index_size': self.index.ntotal,
                'dimension': self.dimension,
                'avg_document_length': np.mean(doc_lengths),
                'max_document_length': max(doc_lengths),
                'min_document_length': min(doc_lengths),
                'avg_question_length': np.mean(question_lengths) if question_lengths else 0,
                'avg_answer_length': np.mean(answer_lengths) if answer_lengths else 0,
                'unique_sources': len(set(meta.get('source', '') for meta in self.metadatas)),
                'model_name': 'all-MiniLM-L6-v2',
                'index_type': type(self.index).__name__
            }
            
            return stats
            
        except Exception as e:
            print(f"❌ Error getting statistics: {e}")
            return {}

    def __del__(self):
        """Cleanup on deletion"""
        try:
            if hasattr(self, '_session') and self._session:
                self._session.close()
        except:
            pass
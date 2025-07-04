import os
import faiss
import pickle
import pandas as pd  # Required for reading CSV
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

class VectorDB_FAISS:
    def __init__(self, index_path: str = "faiss_index"):
        self.index_path = index_path
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.dimension = 384  # MiniLM-L6-v2 has 384 dims
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []  # Store original docs
        self.metadatas = []
        self.ids = []

        self._load_index()

    def _load_index(self):
        index_file = self.index_path + ".index"
        metadata_file = self.index_path + ".pkl"

        if os.path.exists(index_file) and os.path.exists(metadata_file):
            try:
                self.index = faiss.read_index(index_file)
                with open(metadata_file, "rb") as f:
                    data = pickle.load(f)
                    # Handle both old and new pickle formats
                    if isinstance(data, tuple) and len(data) == 3:
                        self.documents, self.metadatas, self.ids = data
                    else:
                        # If old format, try to handle gracefully
                        self.documents = data.get('documents', [])
                        self.metadatas = data.get('metadatas', [])
                        self.ids = data.get('ids', [])
                    
                    print(f"✅ Loaded FAISS index with {len(self.documents)} documents")
                    print(f"📊 Index contains {self.index.ntotal} vectors")
                    
                    # Verify data consistency
                    if len(self.documents) != self.index.ntotal:
                        print(f"⚠️ Warning: Document count ({len(self.documents)}) doesn't match index size ({self.index.ntotal})")
                        
            except Exception as e:
                print(f"❌ Error loading index metadata: {e}")
                self.documents = []
                self.metadatas = []
                self.ids = []
                # Create fresh index
                self.index = faiss.IndexFlatL2(self.dimension)
        else:
            print("ℹ️ No existing FAISS index found. Starting fresh.")

    def _save_index(self):
        try:
            faiss.write_index(self.index, self.index_path + ".index")
            with open(self.index_path + ".pkl", "wb") as f:
                pickle.dump((self.documents, self.metadatas, self.ids), f)
            print("✅ Saved FAISS index and metadata")
        except Exception as e:
            print(f"❌ Error saving index: {e}")

    def add_documents(self, documents: List[str], metadatas: List[Dict], ids: List[str]):
        try:
            # Filter out empty documents
            valid_docs = []
            valid_metas = []
            valid_ids = []
            
            for doc, meta, doc_id in zip(documents, metadatas, ids):
                if doc and doc.strip():  # Only add non-empty documents
                    valid_docs.append(doc.strip())
                    valid_metas.append(meta)
                    valid_ids.append(doc_id)
            
            if not valid_docs:
                print("⚠️ No valid documents to add")
                return
            
            print(f"🔄 Encoding {len(valid_docs)} documents...")
            embeddings = self.model.encode(valid_docs, convert_to_numpy=True).astype("float32")
            print(f"📊 Generated embeddings shape: {embeddings.shape}")
            
            self.index.add(embeddings)
            self.documents.extend(valid_docs)
            self.metadatas.extend(valid_metas)
            self.ids.extend(valid_ids)

            self._save_index()
            print(f"✅ Added {len(valid_docs)} documents to FAISS index")
            
        except Exception as e:
            print(f"❌ Error adding documents: {e}")
            import traceback
            traceback.print_exc()

    def search(self, query: str, n_results: int = 5):
        if len(self.documents) == 0 or self.index.ntotal == 0:
            print("⚠️ No documents in index to search")
            return []

        try:
            print(f"🔍 Searching for: '{query}'")
            print(f"📊 Index has {self.index.ntotal} vectors, {len(self.documents)} documents")
            
            embedding = self.model.encode([query], convert_to_numpy=True).astype("float32")
            print(f"🔢 Query embedding shape: {embedding.shape}")
            
            # Ensure we don't ask for more results than available
            actual_n_results = min(n_results, self.index.ntotal)
            
            D, I = self.index.search(embedding, actual_n_results)
            print(f"🎯 Search returned {len(I[0])} indices")

            results = []
            for i, idx in enumerate(I[0]):
                if idx >= 0 and idx < len(self.documents):  # Valid index
                    result = {
                        "document": self.documents[idx],
                        "metadata": self.metadatas[idx] if idx < len(self.metadatas) else {},
                        "distance": float(D[0][i])
                    }
                    results.append(result)
                    print(f"Result {i+1}: distance={float(D[0][i]):.4f}, doc_length={len(self.documents[idx])}")
                else:
                    print(f"⚠️ Skipped invalid index {idx}")

            print(f"✅ Returning {len(results)} results")
            return results
            
        except Exception as e:
            print(f"❌ Error during search: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _call_deepseek_api(self, prompt: str, context: str = "") -> str:
        """Call DeepSeek API via OpenRouter with optional context"""
        try:
            api_key = os.getenv('OPENROUTER_API_KEY')
            api_url = os.getenv('OPENROUTER_API_URL', 'https://openrouter.ai/api/v1/chat/completions')
            model = os.getenv('DEEPSEEK_MODEL', 'deepseek/deepseek-chat')

            if not api_key:
                return "Error: OpenRouter API key not configured. Please set OPENROUTER_API_KEY in your environment variables."

            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json',
                'HTTP-Referer': os.getenv('YOUR_SITE_URL', 'http://localhost:5000'),
                'X-Title': os.getenv('YOUR_SITE_NAME', 'Flask RAG App')
            }

            # Construct the prompt with context if available
            if context:
                full_prompt = f"""Based on the following context information, please answer the user's question. If the context doesn't contain relevant information, please say so and provide a general answer.

Context:
{context}

User Question: {prompt}

Please provide a helpful and accurate answer:"""
            else:
                full_prompt = prompt

            payload = {
                'model': model,
                'messages': [
                    {
                        'role': 'user',
                        'content': full_prompt
                    }
                ],
                'max_tokens': 1000,
                'temperature': 0.7,
                'stream': False
            }

            print(f"🤖 Calling DeepSeek via OpenRouter...")
            response = requests.post(api_url, headers=headers, json=payload, timeout=30)

            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    content = result['choices'][0]['message']['content']
                    print("✅ DeepSeek API response received via OpenRouter")
                    return content
                else:
                    print("❌ DeepSeek API returned empty response")
                    return "DeepSeek API returned an empty response."
            else:
                print(f"❌ OpenRouter API Error: {response.status_code}")
                print(f"Response: {response.text}")
                return f"Error calling DeepSeek via OpenRouter: {response.status_code} - {response.text}"

        except requests.exceptions.Timeout:
            print("❌ OpenRouter API request timed out")
            return "OpenRouter API request timed out. Please try again."
        except Exception as e:
            print(f"❌ Exception while calling DeepSeek via OpenRouter: {str(e)}")
            return f"Error calling DeepSeek via OpenRouter: {str(e)}"

    def rag_search_and_answer(self, query: str, n_results: int = 3, distance_threshold: float = 1.0) -> Dict:
        """
        Perform RAG (Retrieval-Augmented Generation) search and answer.
        
        Args:
            query: The user's question
            n_results: Number of documents to retrieve
            distance_threshold: Maximum distance for relevant results
            
        Returns:
            Dictionary containing the answer, sources, and metadata
        """
        try:
            print(f"🔍 RAG Search for: '{query}'")
            
            # Step 1: Search vector database
            search_results = self.search(query, n_results)
            
            # Step 2: Filter relevant results
            relevant_results = [r for r in search_results if r['distance'] < distance_threshold]
            
            if relevant_results:
                print(f"📚 Found {len(relevant_results)} relevant documents")
                
                # Step 3: Prepare context from relevant documents
                context_parts = []
                for i, result in enumerate(relevant_results, 1):
                    doc = result['document']
                    # Extract clean context
                    if 'Answer:' in doc:
                        parts = doc.split('Answer:')
                        if len(parts) == 2:
                            question = parts[0].replace('Question:', '').strip()
                            answer = parts[1].strip()
                            context_parts.append(f"Document {i}:\nQ: {question}\nA: {answer}")
                    else:
                        context_parts.append(f"Document {i}:\n{doc}")
                
                context = "\n\n".join(context_parts)
                
                # Step 4: Use DeepSeek with context
                print("🤖 Using DeepSeek with retrieved context")
                answer = self._call_deepseek_api(query, context)
                
                return {
                    'answer': answer,
                    'method': 'rag_with_context',
                    'sources': relevant_results,
                    'context_used': True,
                    'num_sources': len(relevant_results)
                }
            else:
                print("🌐 No relevant context found, using DeepSeek without context")
                answer = self._call_deepseek_api(query)
                
                return {
                    'answer': answer,
                    'method': 'deepseek_only',
                    'sources': [],
                    'context_used': False,
                    'num_sources': 0
                }
                
        except Exception as e:
            print(f"❌ Error in RAG search and answer: {e}")
            import traceback
            traceback.print_exc()
            return {
                'answer': f"Error during RAG processing: {str(e)}",
                'method': 'error',
                'sources': [],
                'context_used': False,
                'num_sources': 0
            }

    def hybrid_search(self, query: str, mode: str = "auto", n_results: int = 3, distance_threshold: float = 1.0) -> Dict:
        """
        Hybrid search with multiple modes.
        
        Args:
            query: The user's question
            mode: "vector_only", "deepseek_only", "rag", or "auto"
            n_results: Number of documents to retrieve
            distance_threshold: Maximum distance for relevant results
            
        Returns:
            Dictionary containing the answer and metadata
        """
        if mode == "vector_only":
            # Only use vector database
            search_results = self.search(query, n_results)
            relevant_results = [r for r in search_results if r['distance'] < distance_threshold]
            
            if relevant_results:
                context = "Based on your data:\n\n"
                for i, result in enumerate(relevant_results, 1):
                    doc = result['document']
                    if 'Answer:' in doc:
                        parts = doc.split('Answer:')
                        if len(parts) == 2:
                            answer = parts[1].strip()
                            context += f"{i}. {answer}\n\n"
                    else:
                        context += f"{i}. {doc}\n\n"
                
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
            # Only use DeepSeek
            answer = self._call_deepseek_api(query)
            return {
                'answer': answer,
                'method': 'deepseek_only',
                'sources': [],
                'context_used': False,
                'num_sources': 0
            }
        
        elif mode == "rag":
            # Always use RAG approach
            return self.rag_search_and_answer(query, n_results, distance_threshold)
        
        else:  # mode == "auto"
            # Auto mode: use vector if relevant results, otherwise DeepSeek
            search_results = self.search(query, n_results)
            relevant_results = [r for r in search_results if r['distance'] < distance_threshold]
            
            if relevant_results:
                # Use RAG approach
                return self.rag_search_and_answer(query, n_results, distance_threshold)
            else:
                # Use DeepSeek only
                answer = self._call_deepseek_api(query)
                return {
                    'answer': answer,
                    'method': 'deepseek_fallback',
                    'sources': [],
                    'context_used': False,
                    'num_sources': 0
                }

    def index_csv(self, csv_path: str):
        if not os.path.exists(csv_path):
            return False, f"CSV path '{csv_path}' does not exist."

        try:
            print(f"📁 Reading CSV from: {csv_path}")
            df = pd.read_csv(csv_path)
            print(f"📊 CSV shape: {df.shape}")
            print(f"📋 CSV columns: {list(df.columns)}")

            # Check for required columns (case-insensitive)
            df_columns_lower = [col.lower() for col in df.columns]
            
            question_col = None
            answer_col = None
            
            for col in df.columns:
                if col.lower() in ['question', 'q']:
                    question_col = col
                elif col.lower() in ['answer', 'a', 'response']:
                    answer_col = col
            
            if not question_col or not answer_col:
                available_cols = ', '.join(df.columns)
                return False, f"CSV must contain 'Question' and 'Answer' columns (case-insensitive). Available columns: {available_cols}"

            # Combine question and answer for better context
            documents = []
            for _, row in df.iterrows():
                question = str(row[question_col]).strip()
                answer = str(row[answer_col]).strip()
                
                if question and answer and question != 'nan' and answer != 'nan':
                    # Create a comprehensive document
                    doc = f"Question: {question}\nAnswer: {answer}"
                    documents.append(doc)

            if not documents:
                return False, "No valid question-answer pairs found in CSV"

            metadatas = []
            for _, row in df.iterrows():
                question = str(row[question_col]).strip()
                answer = str(row[answer_col]).strip()
                
                if question and answer and question != 'nan' and answer != 'nan':
                    meta = {
                        'question': question,
                        'answer': answer,
                        'source': csv_path
                    }
                    # Add any additional columns as metadata
                    for col in df.columns:
                        if col not in [question_col, answer_col]:
                            meta[col] = str(row[col])
                    metadatas.append(meta)

            ids = [f"doc_{i}" for i in range(len(documents))]

            print(f"📝 Prepared {len(documents)} documents for indexing")
            self.add_documents(documents, metadatas, ids)
            
            return True, f"Successfully indexed {len(documents)} Q&A pairs from {csv_path}"

        except Exception as e:
            print(f"❌ Error indexing CSV: {e}")
            import traceback
            traceback.print_exc()
            return False, str(e)

    def get_collection_info(self):
        return {
            "count": len(self.documents),
            "index_size": self.index.ntotal,
            "dimension": self.dimension,
            "index_type": "FAISS L2"
        }

    # Add alias for backward compatibility
    def get_info(self):
        return self.get_collection_info()

    def clear_index(self):
        """Clear the entire index"""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        self.metadatas = []
        self.ids = []
        self._save_index()
        print("🗑️ Cleared FAISS index")

    def get_document_by_id(self, doc_id: str):
        """Get a specific document by ID"""
        try:
            idx = self.ids.index(doc_id)
            return {
                "document": self.documents[idx],
                "metadata": self.metadatas[idx],
                "id": doc_id
            }
        except ValueError:
            return None
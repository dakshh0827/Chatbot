from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
from dotenv import load_dotenv
import json
from vector_db import VectorDB_FAISS
import time
from concurrent.futures import ThreadPoolExecutor
import asyncio
import threading
from functools import lru_cache

load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize FAISS vector database with lazy loading
vector_db = None
db_lock = threading.Lock()

def get_vector_db():
    """Lazy initialization of vector database"""
    global vector_db
    if vector_db is None:
        with db_lock:
            if vector_db is None:  # Double-check pattern
                vector_db = VectorDB_FAISS()
    return vector_db

# Thread pool for concurrent operations
executor = ThreadPoolExecutor(max_workers=4)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        db = get_vector_db()
        return jsonify({
            'status': 'healthy',
            'vector_db_info': db.get_collection_info(),
            'timestamp': time.time()
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': time.time()
        }), 500

@app.route('/upload-csv', methods=['POST'])
def upload_csv():
    """Process and store CSV data in FAISS vector database"""
    try:
        data = request.get_json()
        csv_path = data.get('csv_path', './data/your_data.csv')

        db = get_vector_db()
        success, message = db.index_csv(csv_path)

        return jsonify({
            'success': success,
            'message': message
        })

    except Exception as e:
        print(f"❌ CSV upload error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests with enhanced RAG capabilities and timeout protection"""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        mode = data.get('mode', 'auto')
        n_results = data.get('n_results', 3)
        distance_threshold = data.get('distance_threshold', 1.0)

        if not user_message:
            return jsonify({'error': 'Message is required'}), 400

        print(f"🔍 Chat request - Mode: {mode}, Query: {user_message}")
        
        # Set a timeout for the entire operation
        start_time = time.time()
        timeout_seconds = 25  # Leave 5 seconds buffer for response processing
        
        try:
            # Use the new hybrid search method with timeout
            db = get_vector_db()
            
            # Run with timeout
            def search_with_timeout():
                return db.hybrid_search(
                    query=user_message,
                    mode=mode,
                    n_results=n_results,
                    distance_threshold=distance_threshold
                )
            
            # Submit to thread pool with timeout
            future = executor.submit(search_with_timeout)
            result = future.result(timeout=timeout_seconds)
            
            processing_time = time.time() - start_time
            print(f"⏱️ Processing completed in {processing_time:.2f} seconds")

            return jsonify({
                'response': result['answer'],
                'method': result['method'],
                'context_used': result['context_used'],
                'sources': [r['metadata'] for r in result['sources']] if result['sources'] else [],
                'debug_info': {
                    'mode': mode,
                    'num_sources': result['num_sources'],
                    'vector_db_count': db.get_collection_info()['count'],
                    'processing_time': processing_time
                }
            })
            
        except Exception as search_error:
            print(f"❌ Search timeout or error: {search_error}")
            # Fallback to simple response
            return jsonify({
                'response': "I apologize, but I'm experiencing technical difficulties. Please try again with a simpler question.",
                'method': 'fallback',
                'context_used': False,
                'sources': [],
                'debug_info': {
                    'error': str(search_error),
                    'timeout': True
                }
            })

    except Exception as e:
        print(f"❌ Chat endpoint error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

# Optimized DeepSeek API call with connection pooling
session = requests.Session()
session.mount('https://', requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=20))

@lru_cache(maxsize=100)
def get_api_config():
    """Cached API configuration"""
    return {
        'api_key': os.getenv('OPENROUTER_API_KEY'),
        'api_url': os.getenv('OPENROUTER_API_URL', 'https://openrouter.ai/api/v1/chat/completions'),
        'model': os.getenv('DEEPSEEK_MODEL', 'deepseek/deepseek-chat'),
        'site_url': os.getenv('YOUR_SITE_URL', 'http://localhost:5000'),
        'site_name': os.getenv('YOUR_SITE_NAME', 'Flask RAG App')
    }

def call_deepseek_api(prompt: str) -> str:
    """Optimized DeepSeek API call with connection pooling and shorter timeout"""
    try:
        config = get_api_config()
        
        if not config['api_key']:
            return "Error: OpenRouter API key not configured. Please set OPENROUTER_API_KEY in your environment variables."

        headers = {
            'Authorization': f'Bearer {config["api_key"]}',
            'Content-Type': 'application/json',
            'HTTP-Referer': config['site_url'],
            'X-Title': config['site_name']
        }

        payload = {
            'model': config['model'],
            'messages': [
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'max_tokens': 800,  # Reduced for faster response
            'temperature': 0.5,  # Reduced for more focused responses
            'stream': False
        }

        print(f"🤖 Calling DeepSeek via OpenRouter with query: {prompt[:50]}...")
        
        # Shorter timeout for faster failure
        response = session.post(
            config['api_url'], 
            headers=headers, 
            json=payload, 
            timeout=15  # Reduced from 30 to 15 seconds
        )

        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']
                print("✅ DeepSeek API response received via OpenRouter")
                return content
            else:
                print("❌ DeepSeek API returned empty response")
                return "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
        else:
            print(f"❌ OpenRouter API Error: {response.status_code}")
            return f"I'm experiencing technical difficulties with the AI service. Please try again later."

    except requests.exceptions.Timeout:
        print("❌ OpenRouter API request timed out")
        return "The AI service is taking too long to respond. Please try a simpler question or try again later."
    except requests.exceptions.ConnectionError:
        print("❌ OpenRouter API connection error")
        return "I'm having trouble connecting to the AI service. Please check your internet connection and try again."
    except Exception as e:
        print(f"❌ Exception while calling DeepSeek via OpenRouter: {str(e)}")
        return "I encountered an unexpected error. Please try again with a different question."

@app.route('/search', methods=['POST'])
def search_vector_db():
    """Search FAISS vector database directly with timeout protection"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        n_results = data.get('n_results', 5)

        if not query:
            return jsonify({'error': 'Query is required'}), 400

        db = get_vector_db()
        
        # Use thread pool for search with timeout
        future = executor.submit(db.search, query, n_results)
        results = future.result(timeout=10)  # 10 second timeout

        return jsonify({
            'results': results,
            'query': query
        })

    except Exception as e:
        print(f"❌ Search error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/rag-chat', methods=['POST'])
def rag_chat():
    """Enhanced RAG chat endpoint with timeout protection"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        n_results = data.get('n_results', 3)
        distance_threshold = data.get('distance_threshold', 1.0)
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        db = get_vector_db()
        
        # Use thread pool with timeout
        future = executor.submit(
            db.rag_search_and_answer, 
            message, 
            n_results, 
            distance_threshold
        )
        result = future.result(timeout=20)  # 20 second timeout
        
        return jsonify({
            'response': result['answer'],
            'method': result['method'],
            'context_used': result['context_used'],
            'num_sources': result['num_sources'],
            'sources': [
                {
                    'question': r['metadata'].get('question', ''),
                    'answer': r['metadata'].get('answer', ''),
                    'distance': r['distance']
                } for r in result['sources']
            ]
        })
        
    except Exception as e:
        print(f"❌ RAG chat error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/deepseek-chat', methods=['POST'])
def deepseek_chat():
    """Direct DeepSeek API chat endpoint via OpenRouter with timeout protection"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Use thread pool with timeout
        future = executor.submit(call_deepseek_api, message)
        response = future.result(timeout=15)
        
        return jsonify({
            'response': response,
            'source': 'deepseek-openrouter'
        })
        
    except Exception as e:
        print(f"❌ DeepSeek chat error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# Add cleanup on app shutdown
@app.teardown_appcontext
def cleanup(error):
    """Cleanup resources"""
    pass

if __name__ == '__main__':
    print("🚀 Starting Flask app...")
    
    # Initialize vector DB in main thread
    try:
        db = get_vector_db()
        print(f"📊 Vector DB info: {db.get_collection_info()}")
    except Exception as e:
        print(f"⚠️ Warning: Could not initialize vector DB: {e}")
    
    # Check if OpenRouter API key is configured
    if os.getenv('OPENROUTER_API_KEY'):
        print("✅ OpenRouter API key configured")
        model = os.getenv('DEEPSEEK_MODEL', 'deepseek/deepseek-chat')
        print(f"🤖 Using DeepSeek model: {model}")
    else:
        print("⚠️  OpenRouter API key not found. Please set OPENROUTER_API_KEY in your .env file")
    
    # Production settings
    app.run(
        debug=False,  # Disable debug in production
        host='0.0.0.0', 
        port=int(os.getenv('PORT', 5000)),
        threaded=True  # Enable threading
    )
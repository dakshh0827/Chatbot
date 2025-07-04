from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
from dotenv import load_dotenv
import json
from vector_db import VectorDB_FAISS  # <- Replace VectorDB with FaissDB

load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize FAISS vector database
vector_db = VectorDB_FAISS()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'vector_db_info': vector_db.get_collection_info()  # Fixed method name
    })

@app.route('/upload-csv', methods=['POST'])
def upload_csv():
    """Process and store CSV data in FAISS vector database"""
    try:
        data = request.get_json()
        csv_path = data.get('csv_path', './data/your_data.csv')

        success, message = vector_db.index_csv(csv_path)

        return jsonify({
            'success': success,
            'message': message
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests with enhanced RAG capabilities"""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        mode = data.get('mode', 'auto')  # auto, vector_only, deepseek_only, rag
        n_results = data.get('n_results', 3)
        distance_threshold = data.get('distance_threshold', 1.0)

        if not user_message:
            return jsonify({'error': 'Message is required'}), 400

        print(f"🔍 Chat request - Mode: {mode}, Query: {user_message}")
        
        # Use the new hybrid search method
        result = vector_db.hybrid_search(
            query=user_message,
            mode=mode,
            n_results=n_results,
            distance_threshold=distance_threshold
        )

        return jsonify({
            'response': result['answer'],
            'method': result['method'],
            'context_used': result['context_used'],
            'sources': [r['metadata'] for r in result['sources']] if result['sources'] else [],
            'debug_info': {
                'mode': mode,
                'num_sources': result['num_sources'],
                'vector_db_count': vector_db.get_collection_info()['count']
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def call_deepseek_api(prompt: str) -> str:
    """Call DeepSeek API via OpenRouter"""
    try:
        api_key = os.getenv('OPENROUTER_API_KEY')
        api_url = os.getenv('OPENROUTER_API_URL', 'https://openrouter.ai/api/v1/chat/completions')
        
        # OpenRouter model names for DeepSeek
        model = os.getenv('DEEPSEEK_MODEL', 'deepseek/deepseek-chat')

        if not api_key:
            return "Error: OpenRouter API key not configured. Please set OPENROUTER_API_KEY in your environment variables."

        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'HTTP-Referer': os.getenv('YOUR_SITE_URL', 'http://localhost:5000'),  # Optional: for including your app on openrouter.ai rankings
            'X-Title': os.getenv('YOUR_SITE_NAME', 'Flask RAG App')  # Optional: shows in rankings on openrouter.ai
        }

        payload = {
            'model': model,
            'messages': [
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'max_tokens': 1000,
            'temperature': 0.7,
            'stream': False
        }

        print(f"🤖 Calling DeepSeek via OpenRouter with query: {prompt[:50]}...")
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

def call_grok_api(prompt: str) -> str:
    """Call Grok API with the given prompt (kept for backward compatibility)"""
    try:
        api_key = os.getenv('GROK_API_KEY')
        api_url = os.getenv('GROK_API_URL')

        if not api_key:
            return "Error: Grok API key not configured"

        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

        payload = {
            'model': 'grok-beta',
            'messages': [
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'max_tokens': 1000,
            'temperature': 0.7
        }

        response = requests.post(api_url, headers=headers, json=payload)

        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            print("❌ Grok API Error:", response.status_code, response.text)
            return f"Error calling Grok API: {response.status_code} - {response.text}"

    except Exception as e:
        print("❌ Exception while calling Grok API:", str(e))
        return f"Error calling Grok API: {str(e)}"

@app.route('/search', methods=['POST'])
def search_vector_db():
    """Search FAISS vector database directly"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        n_results = data.get('n_results', 5)

        results = vector_db.search(query, n_results)

        return jsonify({
            'results': results,
            'query': query
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/rag-chat', methods=['POST'])
def rag_chat():
    """Enhanced RAG chat endpoint"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        n_results = data.get('n_results', 3)
        distance_threshold = data.get('distance_threshold', 1.0)
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        result = vector_db.rag_search_and_answer(
            query=message,
            n_results=n_results,
            distance_threshold=distance_threshold
        )
        
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
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/deepseek-chat', methods=['POST'])
def deepseek_chat():
    """Direct DeepSeek API chat endpoint via OpenRouter"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        response = call_deepseek_api(message)
        
        return jsonify({
            'response': response,
            'source': 'deepseek-openrouter'
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        response = call_deepseek_api(message)
        
        return jsonify({
            'response': response,
            'source': 'deepseek-openrouter'
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Flask app...")
    print(f"📊 Vector DB info: {vector_db.get_collection_info()}")
    
    # Check if OpenRouter API key is configured
    if os.getenv('OPENROUTER_API_KEY'):
        print("✅ OpenRouter API key configured")
        model = os.getenv('DEEPSEEK_MODEL', 'deepseek/deepseek-chat')
        print(f"🤖 Using DeepSeek model: {model}")
    else:
        print("⚠️  OpenRouter API key not found. Please set OPENROUTER_API_KEY in your .env file")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
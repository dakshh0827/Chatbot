# 🧠 Company-Specific AI Chatbot

A full-stack, AI-powered chatbot application built with a **React** frontend and a **Flask** backend. This chatbot delivers precise answers by leveraging a **company-specific vector Q&A dataset** and falls back on **DeepSeek via OpenRouter** for general web-based queries.

---

## ✨ Core Features

- **🔍 Hybrid Search:** Prioritizes the internal vector database for company-specific information and seamlessly transitions to web search for broader inquiries.
- **📚 Custom Knowledge Base:** Utilizes a specialized vector dataset to provide accurate and context-aware answers related to your company.
- **🤖 Intelligent Web Fallback:** Integrates the powerful **DeepSeek** model through the OpenRouter API to handle questions outside its primary knowledge base.
- **⚛️ Modern Frontend:** A sleek and responsive user interface built with **React**.
- **🔧 Lightweight Backend:** An efficient and scalable server powered by **Flask**.
- **⚡ Real-time Responses:** Fast query processing with optimized vector similarity search.
- **🔒 Secure API Integration:** Environment-based configuration for API keys and endpoints.

---

## 📦 Tech Stack

| Category        | Technology                                      |
|:----------------|:-----------------------------------------------|
| **Frontend**    | `React`, `Tailwind CSS`, `Axios`              |
| **Backend**     | `Flask`, `Flask-CORS`, `Python-dotenv`        |
| **Vector DB**   | `FAISS`, `Sentence Transformers`              |
| **AI Model**    | `DeepSeek` (via OpenRouter API)               |
| **Languages**   | `Python`, `JavaScript/TypeScript`             |
| **Deployment** | `Docker` (optional), `Gunicorn`               |

---

## 🗂️ Project Structure

```
chatbotAPI/
├── backend/
│   ├── data/
│   │   ├── faiss_index_files/
│   │   │   ├── index.faiss
│   │   │   ├── metadata.json
│   │   │   └── embeddings.pkl
│   │   ├── qa_dataset.json
│   │   └── processed_data/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── vector_search.py
│   │   └── openrouter_client.py
│   ├── routes/
│   │   ├── __init__.py
│   │   └── chat_routes.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_processor.py
│   │   └── config.py
│   ├── app.py
│   ├── requirements.txt
│   ├── .env.example
│   └── README.md
├── frontend/
│   ├── public/
│   │   ├── index.html
│   │   ├── favicon.ico
│   │   └── manifest.json
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatInterface.jsx
│   │   │   ├── MessageBubble.jsx
│   │   │   └── InputField.jsx
│   │   ├── services/
│   │   │   └── api.js
│   │   ├── styles/
│   │   │   └── index.css
│   │   ├── App.js
│   │   └── index.js
│   ├── package.json
│   ├── package-lock.json
│   └── README.md
├── docs/
│   ├── API.md
│   └── DEPLOYMENT.md
├── .gitignore
├── docker-compose.yml
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+**
- **Node.js 16+**
- **npm or yarn**
- **OpenRouter API Key**

### 🐍 Backend Setup (Flask API)

1. **Clone the Repository**
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>/backend
   ```

2. **Create and Activate Virtual Environment**
   ```bash
   # For Windows
   python -m venv venv
   venv\Scripts\activate
   
   # For macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**
   ```bash
   cp .env.example .env
   # Edit .env file with your API keys (see Environment Variables section)
   ```

5. **Initialize Vector Database** (if needed)
   ```bash
   python utils/data_processor.py
   ```

6. **Run the Flask App**
   ```bash
   python app.py
   ```
   The backend will be available at `http://localhost:5000`

### ⚛️ Frontend Setup (React App)

1. **Navigate to the Frontend Directory** (in a new terminal)
   ```bash
   cd ../frontend
   ```

2. **Install Dependencies**
   ```bash
   npm install
   # or
   yarn install
   ```

3. **Start the Development Server**
   ```bash
   npm run dev
   # or
   yarn dev
   ```
   The frontend will be available at `http://localhost:3000`

---

## 🔐 Environment Variables

Create a `.env` file in the `backend/` directory with the following configuration:

```env
# OpenRouter API Configuration
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_API_URL=https://openrouter.ai/api/v1
DEEPSEEK_MODEL=deepseek/deepseek-coder

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True
FLASK_PORT=5000

# Vector Database Configuration
FAISS_INDEX_PATH=./data/faiss_index_files/
SIMILARITY_THRESHOLD=0.75
MAX_RESULTS=5

# CORS Configuration
ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/app.log
```

### Getting Your OpenRouter API Key

1. Visit [OpenRouter.ai](https://openrouter.ai)
2. Sign up for an account
3. Navigate to the API Keys section
4. Generate a new API key
5. Copy the key to your `.env` file

---

## 🔧 How It Works

### Architecture Overview

The chatbot follows a hybrid approach that combines local knowledge with external AI capabilities:

```
User Input → React Frontend → Flask Backend → Vector Search → Response Generation
                                    ↓
                            FAISS Vector Database
                                    ↓
                            Company-specific Q&A
                                    ↓
                         [High Similarity Found?]
                                ↙        ↘
                            YES              NO
                             ↓                ↓
                    Return Internal     Query DeepSeek
                        Answer          via OpenRouter
                             ↓                ↓
                        Display to User ←────┘
```

### Step-by-Step Process

1. **User Input Processing**
   - User submits a question through the React frontend
   - Frontend sends HTTP POST request to Flask backend at `/api/chat`

2. **Vector Similarity Search**
   - Backend processes the query using sentence transformers
   - Converts query to embeddings and searches FAISS index
   - Calculates cosine similarity with existing Q&A pairs

3. **Decision Logic**
   - **High Similarity (≥ 0.75):** Returns pre-defined company answer
   - **Low Similarity (< 0.75):** Forwards query to external API

4. **External AI Integration**
   - If no internal match found, query is sent to DeepSeek model
   - OpenRouter API handles the external request
   - Response is formatted and returned to user

5. **Response Delivery**
   - Generated answer is sent back to React frontend
   - Frontend displays the response in the chat interface

### Key Components

- **Vector Search Engine:** Uses FAISS for efficient similarity search
- **Embedding Model:** Sentence-BERT for semantic understanding
- **Fallback System:** DeepSeek model for general knowledge queries
- **Caching Layer:** Stores frequently accessed embeddings
- **Error Handling:** Graceful fallbacks and user-friendly error messages

---

## 📊 API Endpoints

### Chat Endpoint
```http
POST /api/chat
Content-Type: application/json

{
  "message": "What is our company's return policy?"
}
```

**Response:**
```json
{
  "response": "Our company offers a 30-day return policy...",
  "source": "internal",
  "confidence": 0.85,
  "timestamp": "2025-07-23T10:30:00Z"
}
```

### Health Check
```http
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime": "2 hours, 15 minutes"
}
```

---

## 🎨 Customization

### Adding Company Data

1. **Prepare your Q&A dataset** in JSON format:
   ```json
   [
     {
       "question": "What are your business hours?",
       "answer": "We are open Monday-Friday 9AM-6PM EST."
     }
   ]
   ```

2. **Process and index the data:**
   ```bash
   python utils/data_processor.py --input your_data.json
   ```

3. **Restart the backend** to load the new index

### Frontend Customization

- **Styling:** Modify `src/styles/index.css` for custom themes
- **Components:** Update React components in `src/components/`
- **Branding:** Replace logos and colors in the public directory

---

## 📸 Screenshot

<img width="398" height="588" alt="Chatbot Interface Screenshot" src="https://github.com/user-attachments/assets/354605a5-b4e6-47ad-ab1f-152505438b19" />

---

## 🧪 Testing

### Backend Tests
```bash
cd backend
python -m pytest tests/
```

### Frontend Tests
```bash
cd frontend
npm test
```

### Integration Tests
```bash
# Start both services, then run:
npm run test:e2e
```

---

## 🚀 Deployment

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up --build
```

### Production Deployment
```bash
# Backend with Gunicorn
cd backend
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Frontend build
cd frontend
npm run build
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🆘 Support

- **Documentation:** Check the `docs/` directory for detailed guides
- **Issues:** Report bugs on [GitHub Issues](https://github.com/<your-username>/<repo-name>/issues)
- **Discussions:** Join conversations in [GitHub Discussions](https://github.com/<your-username>/<repo-name>/discussions)

---

## 🙏 Acknowledgments

- **OpenRouter** for providing access to state-of-the-art AI models
- **FAISS** for efficient vector similarity search
- **React** and **Flask** communities for excellent documentation
- **DeepSeek** for powerful language model capabilities

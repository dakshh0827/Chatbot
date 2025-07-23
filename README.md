# 🧠 Company-Specific AI Chatbot

A full-stack, AI-powered chatbot application built with a **React** frontend and a **Flask** backend. This chatbot delivers precise answers by leveraging a **company-specific vector Q&A dataset** and falls back on **DeepSeek via OpenRouter** for general web-based queries.

---

## ✨ Core Features

- **🔍 Hybrid Search:** Prioritizes the internal vector database for company-specific information and seamlessly transitions to a web search for broader inquiries.
- **📚 Custom Knowledge Base:** Utilizes a specialized vector dataset to provide accurate and context-aware answers related to your company.
- **🤖 Intelligent Web Fallback:** Integrates the powerful **DeepSeek** model through the OpenRouter API to handle questions outside its primary knowledge base.
- **⚛️ Modern Frontend:** A sleek and responsive user interface built with **React**.
- **🔧 Lightweight Backend:** An efficient and scalable server powered by **Flask**.

---

## 📦 Tech Stack

| Category      | Technology                                    |
| :------------ | :-------------------------------------------- |
| **Frontend**  | `React`, `Tailwind CSS` (optional)           |
| **Backend**   | `Flask`, `FAISS`                             |
| **AI Model**  | `DeepSeek` (via OpenRouter API)              |
| **Vector DB** | `FAISS`                                      |
| **Languages** | `Python`, `JavaScript`                       |

---

## 🗂️ Project Structure

```
chatbotAPI/
├── backend/
│   ├── data/
│   │   └── faiss_index_files/
│   ├── app.py
│   ├── requirements.txt
│   └── .env
└── frontend/
    ├── public/
    ├── src/
    └── package.json
```

---

## 🚀 Getting Started

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
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**
   Create a `.env` file in the `backend` directory and add your API keys.

5. **Run the Flask App**
   ```bash
   python app.py
   ```

### ⚛️ Frontend Setup (React App)

1. **Navigate to the Frontend Directory** (in a new terminal)
   ```bash
   cd ../frontend
   ```

2. **Install Dependencies**
   ```bash
   npm install
   ```

3. **Start the Development Server**
   ```bash
   npm run dev
   ```

---

## 🔐 Environment Variables

Create a `.env` file in the `backend/` directory with the following content:

```env
# backend/.env
OPENROUTER_API_KEY="your_openrouter_api_key_here"
OPENROUTER_API_URL="https://openrouter.ai/api/v1"
DEEPSEEK_MODEL="deepseek/deepseek-coder"
```

---

## 🧪 How It Works

1. **User Input:** The user sends a message through the React frontend.
2. **Backend Processing:** The Flask backend receives the query and first searches the **FAISS vector database** for a relevant, company-specific question.
3. **Hybrid Response Generation:**
   - **Internal Match:** If a high-similarity match is found, the chatbot returns the pre-defined, trusted answer.
   - **Web Fallback:** If no relevant internal answer is found, the query is forwarded to the **DeepSeek model via OpenRouter** for a web-based response.
4. **Display Answer:** The generated answer is sent back to the frontend and displayed to the user.

---

## 📸 Screenshot

<img width="398" height="588" alt="image" src="https://github.com/user-attachments/assets/354605a5-b4e6-47ad-ab1f-152505438b19" />


*Thankyou*

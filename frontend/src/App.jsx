// src/App.jsx
import React from 'react';
import ChatWidget from './components/Chatbot.jsx';

function App() {
  return (
    <div className="min-h-screen bg-gray-100 text-gray-900">
      <header className="p-6 bg-white shadow">
        <h1 className="text-2xl font-bold">Welcome to Jagan Lamps Chat Support</h1>
      </header>

      <main className="p-6">
        <p className="text-lg">Feel free to explore the site. Click the chat icon below if you need help!</p>
      </main>

      {/* Chatbot widget mounted globally */}
      <ChatWidget />
    </div>
  );
}

export default App;

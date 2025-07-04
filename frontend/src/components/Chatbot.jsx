import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { SendHorizonal, MessageCircle, Database, Globe, X, Minimize2 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

const ChatWidget = () => {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const toggleOpen = () => setOpen(!open);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMsg = {
      sender: 'user',
      text: input.trim(),
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setMessages((prev) => [...prev, userMsg]);
    const currentInput = input.trim();
    setInput('');
    setLoading(true);

    try {
      const res = await fetch('http://localhost:5000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: currentInput,
        }),
      });

      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }

      const response = await res.json();
      const botReply = response?.response || 'No response received.';
      const contextUsed = response?.context_used || false;
      const sources = response?.sources || [];
      const debugInfo = response?.debug_info || {};

      const botMsg = {
        sender: 'bot',
        text: botReply,
        contextUsed,
        sources,
        debugInfo,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };

      setMessages((prev) => [...prev, botMsg]);
    } catch (err) {
      console.error('Chat error:', err);
      const errorMsg = {
        sender: 'bot',
        text: '⚠️ Sorry, I encountered an error. Please try again.',
        isError: true,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setLoading(false);
    }
  };

    const formatMessage = (text) => {
    return (
        <div className="prose prose-sm max-w-none">
        <ReactMarkdown
            components={{
            strong: ({ children }) => (
                <strong className="font-semibold text-black">{children}</strong>
            ),
            li: ({ children }) => <li className="ml-4 list-disc">{children}</li>,
            p: ({ children }) => <p className="mb-2">{children}</p>,
            }}
        >
            {text}
        </ReactMarkdown>
        </div>
    );
    };

  return (
    <div className="fixed bottom-6 right-6 z-50">
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, scale: 0.8, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.8, y: 20 }}
            transition={{ duration: 0.2 }}
            className="w-96 h-[32rem] bg-white shadow-2xl rounded-2xl flex flex-col overflow-hidden border"
          >
            {/* Header */}
            <div className="bg-gradient-to-r from-blue-600 to-blue-700 text-white px-4 py-3 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <MessageCircle size={20} />
                <span className="font-semibold">AI Assistant</span>
              </div>
              <button
                onClick={toggleOpen}
                className="hover:bg-blue-800 p-1 rounded-full transition-colors"
              >
                <X size={18} />
              </button>
            </div>

            {/* Messages */}
            <div className="flex-1 p-4 space-y-3 overflow-y-auto bg-gray-50">
              {messages.length === 0 && (
                <div className="text-center text-gray-500 mt-8">
                  <MessageCircle size={48} className="mx-auto mb-4 text-gray-300" />
                  <p className="text-sm">Start a conversation!</p>
                  <p className="text-xs mt-1">I can help you with questions from our database or search the web.</p>
                </div>
              )}

              {messages.map((msg, idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3 }}
                  className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-[85%] px-4 py-3 rounded-2xl ${
                      msg.sender === 'user'
                        ? 'bg-blue-600 text-white rounded-br-md'
                        : msg.isError
                        ? 'bg-red-100 text-red-800 rounded-bl-md border border-red-200'
                        : 'bg-white text-gray-800 rounded-bl-md shadow-sm border'
                    }`}
                  >
                    <div className="text-sm leading-relaxed">
                      {msg.sender === 'user' ? msg.text : formatMessage(msg.text)}
                    </div>

                    {/* Source indicator and metadata */}
                    {msg.sender === 'bot' && !msg.isError && (
                      <div className="mt-2 pt-2 border-t border-gray-200">
                        <div className="flex items-center gap-2 text-xs text-gray-500">
                          {msg.contextUsed ? (
                            <div className="flex items-center gap-1">
                              <Database size={12} className="text-green-600" />
                              <span className="text-green-600 font-medium">From Database</span>
                            </div>
                          ) : (
                            <div className="flex items-center gap-1">
                              <Globe size={12} className="text-blue-600" />
                              <span className="text-blue-600 font-medium">Web Search</span>
                            </div>
                          )}
                          <span className="text-gray-400">•</span>
                          <span>{msg.timestamp}</span>
                        </div>

                        {/* Debug info */}
                        {msg.debugInfo && (
                          <div className="mt-1 text-xs text-gray-400">
                            {msg.debugInfo.total_results > 0 && (
                              <span>
                                {msg.debugInfo.relevant_results}/{msg.debugInfo.total_results} relevant
                              </span>
                            )}
                          </div>
                        )}
                      </div>
                    )}

                    {msg.sender === 'user' && (
                      <div className="mt-1 text-xs text-blue-200">
                        {msg.timestamp}
                      </div>
                    )}
                  </div>
                </motion.div>
              ))}

              {loading && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="flex justify-start"
                >
                  <div className="bg-white px-4 py-3 rounded-2xl rounded-bl-md shadow-sm border">
                    <div className="flex items-center gap-2 text-gray-500">
                      <div className="flex gap-1">
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-100"></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-200"></div>
                      </div>
                      <span className="text-sm">Thinking...</span>
                    </div>
                  </div>
                </motion.div>
              )}

              <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div className="p-4 border-t bg-white">
              <div className="flex items-center gap-2 bg-gray-50 rounded-xl border focus-within:border-blue-500 focus-within:ring-2 focus-within:ring-blue-500/20">
                <input
                  type="text"
                  className="flex-1 px-4 py-3 bg-transparent text-sm focus:outline-none placeholder-gray-500"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && !loading && sendMessage()}
                  disabled={loading}
                  placeholder="Ask me anything..."
                />
                <button
                  onClick={sendMessage}
                  disabled={loading || !input.trim()}
                  className={`p-2 m-2 rounded-full transition-all ${
                    loading || !input.trim()
                      ? 'bg-gray-300 cursor-not-allowed'
                      : 'bg-blue-600 hover:bg-blue-700 text-white shadow-sm hover:shadow-md'
                  }`}
                >
                  <SendHorizonal size={18} />
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Toggle Button */}
      <motion.button
        onClick={toggleOpen}
        className="w-14 h-14 rounded-full bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white shadow-xl flex items-center justify-center transition-all duration-200 hover:shadow-2xl"
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        <AnimatePresence mode="wait">
          {open ? (
            <motion.div
              key="minimize"
              initial={{ rotate: 180, opacity: 0 }}
              animate={{ rotate: 0, opacity: 1 }}
              exit={{ rotate: 180, opacity: 0 }}
              transition={{ duration: 0.2 }}
            >
              <Minimize2 size={24} />
            </motion.div>
          ) : (
            <motion.div
              key="message"
              initial={{ rotate: -180, opacity: 0 }}
              animate={{ rotate: 0, opacity: 1 }}
              exit={{ rotate: -180, opacity: 0 }}
              transition={{ duration: 0.2 }}
            >
              <MessageCircle size={24} />
            </motion.div>
          )}
        </AnimatePresence>
      </motion.button>
    </div>
  );
};

export default ChatWidget;

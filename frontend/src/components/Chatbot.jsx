import React, { useState, useRef, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { SendHorizonal, MessageCircle, Database, Globe, X, Minimize2, AlertCircle, RotateCcw } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

const ChatWidget = () => {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [loadingStage, setLoadingStage] = useState('');
  const [connectionStatus, setConnectionStatus] = useState('online');
  const messagesEndRef = useRef(null);
  const abortControllerRef = useRef(null);

  const toggleOpen = () => setOpen(!open);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Check backend connectivity
  const checkConnection = useCallback(async () => {
    try {
      const response = await fetch('http://localhost:5000/health', {
        method: 'GET',
        timeout: 5000,
      });
      setConnectionStatus(response.ok ? 'online' : 'slow');
    } catch (error) {
      setConnectionStatus('offline');
    }
  }, []);

  useEffect(() => {
    checkConnection();
    const interval = setInterval(checkConnection, 30000); // Check every 30 seconds
    return () => clearInterval(interval);
  }, [checkConnection]);

  const cancelRequest = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setLoading(false);
    setLoadingStage('');
  };

  const retryMessage = async (messageText) => {
    await sendMessage(messageText, true);
  };

  const sendMessage = async (messageText = null, isRetry = false) => {
    const currentInput = messageText || input.trim();
    if (!currentInput || loading) return;

    // Cancel any existing request
    cancelRequest();

    if (!isRetry) {
      const userMsg = {
        id: Date.now(),
        sender: 'user',
        text: currentInput,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setMessages((prev) => [...prev, userMsg]);
      setInput('');
    }

    setLoading(true);
    setLoadingStage('Connecting...');

    // Create new AbortController for this request
    abortControllerRef.current = new AbortController();

    // Set up progressive timeout
    const timeouts = {
      connection: setTimeout(() => setLoadingStage('Searching database...'), 2000),
      processing: setTimeout(() => setLoadingStage('Generating response...'), 8000),
      finalWarning: setTimeout(() => setLoadingStage('Taking longer than expected...'), 15000),
    };

    try {
      const startTime = Date.now();
      
      const response = await fetch('http://localhost:5000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: currentInput,
          mode: 'auto',
          n_results: 3,
          distance_threshold: 1.0,
        }),
        signal: abortControllerRef.current.signal,
        // Set a reasonable timeout (25 seconds to match backend)
        timeout: 25000,
      });

      // Clear all timeouts
      Object.values(timeouts).forEach(clearTimeout);

      const responseTime = Date.now() - startTime;

      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      const botReply = data?.response || 'No response received.';
      const contextUsed = data?.context_used || false;
      const sources = data?.sources || [];
      const debugInfo = data?.debug_info || {};
      const method = data?.method || 'unknown';

      const botMsg = {
        id: Date.now() + 1,
        sender: 'bot',
        text: botReply,
        contextUsed,
        sources,
        debugInfo: {
          ...debugInfo,
          responseTime: responseTime,
          method: method,
        },
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        canRetry: false,
      };

      setMessages((prev) => [...prev, botMsg]);
      setConnectionStatus('online');

    } catch (err) {
      Object.values(timeouts).forEach(clearTimeout);
      
      console.error('Chat error:', err);
      
      let errorMessage = '⚠️ Sorry, I encountered an error.';
      let canRetry = true;
      
      if (err.name === 'AbortError') {
        errorMessage = '⏹️ Request was cancelled.';
        canRetry = false;
      } else if (err.message.includes('fetch')) {
        errorMessage = '🔌 Connection error. Please check your internet connection.';
        setConnectionStatus('offline');
      } else if (err.message.includes('timeout') || err.message.includes('408')) {
        errorMessage = '⏱️ Request timed out. The server might be busy.';
        setConnectionStatus('slow');
      } else if (err.message.includes('500')) {
        errorMessage = '🔧 Server error. Please try again in a moment.';
      } else if (err.message.includes('404')) {
        errorMessage = '🔍 Service not found. Please contact support.';
      }

      const errorMsg = {
        id: Date.now() + 2,
        sender: 'bot',
        text: errorMessage,
        isError: true,
        canRetry: canRetry,
        originalMessage: currentInput,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };

      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setLoading(false);
      setLoadingStage('');
      abortControllerRef.current = null;
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
            code: ({ children }) => (
              <code className="bg-gray-100 px-1 py-0.5 rounded text-sm">{children}</code>
            ),
          }}
        >
          {text}
        </ReactMarkdown>
      </div>
    );
  };

  const ConnectionStatusIndicator = () => {
    const statusConfig = {
      online: { color: 'text-green-500', text: 'Online', icon: '●' },
      slow: { color: 'text-yellow-500', text: 'Slow', icon: '●' },
      offline: { color: 'text-red-500', text: 'Offline', icon: '●' },
    };
    
    const config = statusConfig[connectionStatus];
    
    return (
      <div className={`flex items-center gap-1 text-xs ${config.color}`}>
        <span>{config.icon}</span>
        <span>{config.text}</span>
      </div>
    );
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      cancelRequest();
    };
  }, []);

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
                <div className="flex flex-col">
                  <span className="font-semibold">AI Assistant</span>
                  <ConnectionStatusIndicator />
                </div>
              </div>
              <div className="flex items-center gap-2">
                {loading && (
                  <button
                    onClick={cancelRequest}
                    className="hover:bg-blue-800 p-1 rounded-full transition-colors"
                    title="Cancel request"
                  >
                    <X size={16} />
                  </button>
                )}
                <button
                  onClick={toggleOpen}
                  className="hover:bg-blue-800 p-1 rounded-full transition-colors"
                >
                  <X size={18} />
                </button>
              </div>
            </div>

            {/* Messages */}
            <div className="flex-1 p-4 space-y-3 overflow-y-auto bg-gray-50">
              {messages.length === 0 && (
                <div className="text-center text-gray-500 mt-8">
                  <MessageCircle size={48} className="mx-auto mb-4 text-gray-300" />
                  <p className="text-sm">Start a conversation!</p>
                  <p className="text-xs mt-1">I can search our database or the web to help you.</p>
                  {connectionStatus === 'offline' && (
                    <div className="mt-3 p-2 bg-red-50 border border-red-200 rounded-lg">
                      <div className="flex items-center gap-2 text-red-600 text-xs">
                        <AlertCircle size={14} />
                        <span>Backend service is offline</span>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {messages.map((msg) => (
                <motion.div
                  key={msg.id}
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
                        ? 'bg-red-50 text-red-800 rounded-bl-md border border-red-200'
                        : 'bg-white text-gray-800 rounded-bl-md shadow-sm border'
                    }`}
                  >
                    <div className="text-sm leading-relaxed">
                      {msg.sender === 'user' ? msg.text : formatMessage(msg.text)}
                    </div>

                    {/* Retry button for errors */}
                    {msg.isError && msg.canRetry && (
                      <div className="mt-2 pt-2 border-t border-red-200">
                        <button
                          onClick={() => retryMessage(msg.originalMessage)}
                          disabled={loading}
                          className="flex items-center gap-1 text-xs text-red-600 hover:text-red-800 disabled:opacity-50"
                        >
                          <RotateCcw size={12} />
                          <span>Retry</span>
                        </button>
                      </div>
                    )}

                    {/* Source indicator and metadata */}
                    {msg.sender === 'bot' && !msg.isError && (
                      <div className="mt-2 pt-2 border-t border-gray-200">
                        <div className="flex items-center gap-2 text-xs text-gray-500">
                          {msg.contextUsed ? (
                            <div className="flex items-center gap-1">
                              <Database size={12} className="text-green-600" />
                              <span className="text-green-600 font-medium">Database</span>
                            </div>
                          ) : (
                            <div className="flex items-center gap-1">
                              <Globe size={12} className="text-blue-600" />
                              <span className="text-blue-600 font-medium">Web</span>
                            </div>
                          )}
                          <span className="text-gray-400">•</span>
                          <span>{msg.timestamp}</span>
                          {msg.debugInfo?.responseTime && (
                            <>
                              <span className="text-gray-400">•</span>
                              <span>{(msg.debugInfo.responseTime / 1000).toFixed(1)}s</span>
                            </>
                          )}
                        </div>

                        {/* Method indicator */}
                        {msg.debugInfo?.method && (
                          <div className="mt-1 text-xs text-gray-400">
                            Method: {msg.debugInfo.method}
                            {msg.debugInfo.num_sources > 0 && (
                              <span> • {msg.debugInfo.num_sources} sources</span>
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
                  <div className="bg-white px-4 py-3 rounded-2xl rounded-bl-md shadow-sm border max-w-[85%]">
                    <div className="flex items-center gap-2 text-gray-500">
                      <div className="flex gap-1">
                        <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce"></div>
                        <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce delay-100"></div>
                        <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce delay-200"></div>
                      </div>
                      <div className="flex flex-col">
                        <span className="text-sm font-medium">Processing...</span>
                        {loadingStage && (
                          <span className="text-xs text-gray-400">{loadingStage}</span>
                        )}
                      </div>
                    </div>
                    <button
                      onClick={cancelRequest}
                      className="mt-2 text-xs text-red-500 hover:text-red-700"
                    >
                      Cancel
                    </button>
                  </div>
                </motion.div>
              )}

              <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div className="p-4 border-t bg-white">
              {connectionStatus === 'offline' && (
                <div className="mb-2 p-2 bg-red-50 border border-red-200 rounded-lg">
                  <div className="flex items-center gap-2 text-red-600 text-xs">
                    <AlertCircle size={14} />
                    <span>Service unavailable. Check your connection.</span>
                  </div>
                </div>
              )}
              
              <div className="flex items-center gap-2 bg-gray-50 rounded-xl border focus-within:border-blue-500 focus-within:ring-2 focus-within:ring-blue-500/20">
                <input
                  type="text"
                  className="flex-1 px-4 py-3 bg-transparent text-sm focus:outline-none placeholder-gray-500 disabled:text-gray-400"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && !loading && connectionStatus !== 'offline' && sendMessage()}
                  disabled={loading || connectionStatus === 'offline'}
                  placeholder={connectionStatus === 'offline' ? 'Service unavailable...' : 'Ask me anything...'}
                />
                <button
                  onClick={() => sendMessage()}
                  disabled={loading || !input.trim() || connectionStatus === 'offline'}
                  className={`p-2 m-2 rounded-full transition-all ${
                    loading || !input.trim() || connectionStatus === 'offline'
                      ? 'bg-gray-300 cursor-not-allowed'
                      : 'bg-blue-600 hover:bg-blue-700 text-white shadow-sm hover:shadow-md'
                  }`}
                  title={connectionStatus === 'offline' ? 'Service unavailable' : 'Send message'}
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
        className={`w-14 h-14 rounded-full text-white shadow-xl flex items-center justify-center transition-all duration-200 hover:shadow-2xl ${
          connectionStatus === 'offline'
            ? 'bg-gradient-to-r from-red-500 to-red-600 hover:from-red-600 hover:to-red-700'
            : connectionStatus === 'slow'
            ? 'bg-gradient-to-r from-yellow-500 to-yellow-600 hover:from-yellow-600 hover:to-yellow-700'
            : 'bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800'
        }`}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        title={`AI Assistant (${connectionStatus})`}
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
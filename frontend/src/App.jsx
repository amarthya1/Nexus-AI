const { useState, useRef, useEffect, useCallback } = React;

/* ─────────────────────────────────────────────
   Constants
───────────────────────────────────────────── */
const API_BASE_URL = 'https://nexus-ai-1-8osh.onrender.com';

const MODEL_META = {
  'OpenAI / gpt-4o-mini':                    { label: 'via GPT-4o mini',       color: 'text-sky-400',      bg: 'bg-sky-400/10',      dot: 'bg-sky-400',      icon: '✦' },
  'OpenAI / gpt-4o-mini (fallback)':          { label: 'via GPT-4o mini ⟲',    color: 'text-sky-400',      bg: 'bg-sky-400/10',      dot: 'bg-sky-400',      icon: '✦' },
  'Groq / llama-3.1-70b-versatile':           { label: 'via Llama 3.1 70B',    color: 'text-emerald-400',  bg: 'bg-emerald-400/10',  dot: 'bg-emerald-400',  icon: '⚡' },
  'Groq / llama-3.1-70b-versatile (fallback)':{ label: 'via Llama 3.1 70B ⟲',  color: 'text-emerald-400',  bg: 'bg-emerald-400/10',  dot: 'bg-emerald-400',  icon: '⚡' },
};

const INTENT_META = {
  CODE:     { label: 'Code',     color: 'text-emerald-400' },
  CREATIVE: { label: 'Creative', color: 'text-sky-400'     },
  GENERAL:  { label: 'General',  color: 'text-violet-400'  },
};

const WELCOME_SUGGESTIONS = [
  'Explain how async/await works in Python',
  'Write a short poem about the cosmos',
  'What is the capital of Iceland?',
  'Debug this: for i in range(10) print(i)',
];

/* ─────────────────────────────────────────────
   Helper: format plain text to simple HTML
───────────────────────────────────────────── */
function escapeHtml(str) {
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function formatText(text) {
  text = text.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) =>
    `<pre class="my-3 rounded-xl bg-surface-950 border border-surface-600 p-4 overflow-x-auto text-sm font-mono text-gray-200 leading-relaxed"><code>${escapeHtml(code.trim())}</code></pre>`
  );
  text = text.replace(/`([^`]+)`/g, '<code class="px-1.5 py-0.5 rounded bg-surface-700 text-accent-light text-sm font-mono">$1</code>');
  text = text.replace(/\*\*(.+?)\*\*/g, '<strong class="font-semibold text-white">$1</strong>');
  text = text.replace(/\*(.+?)\*/g, '<em class="italic text-gray-300">$1</em>');
  
  const paras = text.split(/\n\n+/);
  return paras
    .map(p => {
      const lines = p.split('\n');
      if (lines.every(l => l.trim().startsWith('- '))) {
        const items = lines.map(l => `<li class="ml-4 list-disc text-gray-300">${l.replace(/^- /, '')}</li>`).join('');
        return `<ul class="my-1 space-y-1">${items}</ul>`;
      }
      return `<p class="text-gray-300 leading-relaxed">${lines.join('<br/>')}</p>`;
    })
    .join('');
}

/* ─────────────────────────────────────────────
   Sub-components
───────────────────────────────────────────── */
function TypingIndicator() {
  return (
    <div className="flex items-end gap-3 animate-fade-up">
      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-accent to-blue-500 flex items-center justify-center text-xs font-bold shadow-lg">
        AI
      </div>
      <div className="flex items-center gap-1.5 px-4 py-3 rounded-2xl rounded-bl-sm bg-surface-800 border border-surface-600 shadow-lg">
        <span className="w-2 h-2 rounded-full bg-accent animate-pulse inline-block" />
        <span className="w-2 h-2 rounded-full bg-accent animate-pulse inline-block" />
        <span className="w-2 h-2 rounded-full bg-accent animate-pulse inline-block" />
      </div>
    </div>
  );
}

function ModelTag({ modelUsed, intent }) {
  const meta  = MODEL_META[modelUsed]  || { label: modelUsed, color: 'text-gray-400', bg: 'bg-gray-400/10', dot: 'bg-gray-400', icon: '·' };
  const iMeta = INTENT_META[intent]    || { label: intent,    color: 'text-gray-400' };
  return (
    <div className="flex items-center gap-2 mt-2 flex-wrap">
      <span className={`inline-flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full ${meta.bg} ${meta.color} font-medium border border-current/20`}>
        <span className={`w-1.5 h-1.5 rounded-full ${meta.dot}`} />
        {meta.icon} {meta.label}
      </span>
      {intent && (
        <span className={`inline-flex items-center gap-1 text-xs px-2.5 py-1 rounded-full bg-white/5 ${iMeta.color} font-medium border border-current/20`}>
          {iMeta.label} intent
        </span>
      )}
    </div>
  );
}

function ChatMessage({ msg }) {
  const isUser = msg.role === 'user';

  if (isUser) {
    return (
      <div className="flex items-end justify-end gap-3 animate-fade-up">
        <div className="max-w-[75%]">
          <div className="px-4 py-3 rounded-2xl rounded-br-sm bg-gradient-to-br from-accent to-blue-600 text-white shadow-lg shadow-accent/20">
            <p className="text-sm leading-relaxed whitespace-pre-wrap">{msg.content}</p>
          </div>
        </div>
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-gray-600 to-gray-700 border border-surface-600 flex items-center justify-center text-xs font-bold">
          U
        </div>
      </div>
    );
  }

  return (
    <div className="flex items-end gap-3 animate-fade-up">
      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-accent to-blue-500 flex items-center justify-center text-xs font-bold shadow-lg">
        AI
      </div>
      <div className="max-w-[80%]">
        <div className="px-4 py-3 rounded-2xl rounded-bl-sm bg-surface-800 border border-surface-600 shadow-lg">
          <div
            className="text-sm prose-invert"
            dangerouslySetInnerHTML={{ __html: formatText(msg.content) }}
          />
        </div>
        {msg.modelUsed && <ModelTag modelUsed={msg.modelUsed} intent={msg.intent} />}
      </div>
    </div>
  );
}

function SuggestionChip({ text, onSelect }) {
  return (
    <button
      onClick={() => onSelect(text)}
      className="px-3 py-2 rounded-xl bg-surface-800 border border-surface-600 text-xs text-gray-400 hover:text-white hover:border-accent/50 hover:bg-surface-700 transition-all duration-200 text-left"
    >
      {text}
    </button>
  );
}

/* ─────────────────────────────────────────────
   Main App
───────────────────────────────────────────── */
function App() {
  const [messages,  setMessages]  = useState([]);   
  const [input,     setInput]     = useState('');
  const [loading,   setLoading]   = useState(false);
  const [error,     setError]     = useState(null);
  const [sessionId, setSessionId] = useState(() => localStorage.getItem('chat_session_id') || null);

  useEffect(() => {
    if (sessionId) {
      localStorage.setItem('chat_session_id', sessionId);
    }
  }, [sessionId]);

  const bottomRef  = useRef(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  useEffect(() => { textareaRef.current?.focus(); }, []);

  const handleSend = useCallback(async (overrideText) => {
    const text = (overrideText !== undefined ? overrideText : input).trim();
    if (!text || loading) return;

    setError(null);
    setInput('');

    const userMsg   = { role: 'user', content: text };
    const newHistory = [...messages, userMsg];
    setMessages(newHistory);
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE_URL}/api/chat`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({
          messages: newHistory.map(({ role, content }) => ({ role, content })),
          session_id: sessionId
        }),
      });

      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `Server error ${res.status}`);
      }

      const data = await res.json();
      
      if (data.session_id && data.session_id !== sessionId) {
        setSessionId(data.session_id);
      }

      setMessages(prev => [
        ...prev,
        {
          role:      'assistant',
          content:   data.response,
          modelUsed: data.model_used,
          intent:    data.intent,
        },
      ]);
    } catch (err) {
      setError(err.message);
      setMessages(prev => prev.slice(0, -1));
    } finally {
      setLoading(false);
      setTimeout(() => textareaRef.current?.focus(), 50);
    }
  }, [input, messages, loading, sessionId]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const isEmpty = messages.length === 0;

  return (
    <div className="flex flex-col h-screen overflow-hidden">
      {/* Top bar */}
      <header className="flex-shrink-0 flex items-center justify-between px-6 py-3 bg-surface-900/80 border-b border-surface-700 backdrop-blur-sm z-10">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-accent to-blue-500 flex items-center justify-center shadow-lg shadow-accent/30">
            <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
              <path strokeLinecap="round" strokeLinejoin="round" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
            </svg>
          </div>
          <div>
            <h1 className="text-sm font-semibold brand-gradient">AI Chat</h1>
            <p className="text-xs text-gray-500">Smart Router · Llama 3.1 · GPT-4o mini</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <span className="text-xs text-gray-500">Backend live</span>
        </div>
      </header>

      {/* Chat area */}
      <main className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-2xl mx-auto space-y-5">
          {isEmpty && (
            <div className="flex flex-col items-center justify-center pt-16 pb-8 text-center animate-fade-up">
              <div className="w-20 h-20 rounded-3xl bg-gradient-to-br from-accent to-blue-500 flex items-center justify-center shadow-2xl shadow-accent/30 mb-6">
                <svg className="w-10 h-10 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1.5">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z" />
                </svg>
              </div>
              <h2 className="text-2xl font-bold brand-gradient mb-2">How can I help you?</h2>
              <p className="text-gray-500 text-sm max-w-xs mb-8">
                I automatically route your question to the best AI — Llama 3.1 or GPT-4o mini.
              </p>
              <div className="grid grid-cols-2 gap-2 w-full max-w-md">
                {WELCOME_SUGGESTIONS.map(s => (
                  <SuggestionChip key={s} text={s} onSelect={(t) => handleSend(t)} />
                ))}
              </div>
            </div>
          )}

          {messages.map((msg, i) => (
            <ChatMessage key={i} msg={msg} />
          ))}

          {loading && <TypingIndicator />}

          {error && (
            <div className="flex items-center gap-3 px-4 py-3 rounded-2xl bg-red-500/10 border border-red-500/30 text-red-400 text-sm animate-fade-up">
              <svg className="w-4 h-4 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
              </svg>
              <span>{error}</span>
              <button onClick={() => setError(null)} className="ml-auto text-red-400/60 hover:text-red-400">✕</button>
            </div>
          )}
          <div ref={bottomRef} />
        </div>
      </main>

      {/* Input area */}
      <footer className="flex-shrink-0 border-t border-surface-700 bg-surface-900/80 backdrop-blur-sm px-4 py-4">
        <div className="max-w-2xl mx-auto">
          <div className="relative flex items-end gap-3 bg-surface-800 rounded-2xl border border-surface-600 px-4 py-3 shadow-xl focus-within:border-accent/60 transition-all duration-200">
            <textarea
              id="chat-input"
              ref={textareaRef}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask anything… (Shift+Enter for new line)"
              disabled={loading}
              rows={1}
              className="flex-1 bg-transparent resize-none outline-none text-sm text-gray-200 placeholder-gray-600 max-h-40 leading-relaxed disabled:opacity-50"
              style={{ overflowY: input.split('\n').length > 5 ? 'auto' : 'hidden', minHeight: '24px' }}
            />
            <button
              id="send-button"
              onClick={() => handleSend()}
              disabled={!input.trim() || loading}
              className="flex-shrink-0 w-9 h-9 rounded-xl bg-gradient-to-br from-accent to-blue-600 flex items-center justify-center shadow-lg shadow-accent/30 hover:shadow-accent/50 hover:scale-105 active:scale-95 transition-all duration-150 disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:scale-100 disabled:shadow-none"
            >
              {loading
                ? <svg className="w-4 h-4 text-white animate-spin" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/></svg>
                : <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2.5"><path strokeLinecap="round" strokeLinejoin="round" d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.269 20.876L5.999 12zm0 0h7.5" /></svg>
              }
            </button>
          </div>
          <p className="text-center text-gray-600 text-xs mt-2">
            Enter to send · Shift+Enter for new line · Powered by Llama 3.1 &amp; GPT-4o mini
          </p>
        </div>
      </footer>
    </div>
  );
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
import { useState, useRef, useEffect } from 'react'
import axios from 'axios'
import ReactMarkdown from 'react-markdown'
import { Send, Bot, User, MessageSquare } from 'lucide-react'
import { CitationHighlight, CitationList } from './CitationHighlight'
import HallucinationBadge from './HallucinationBadge'

const SUGGESTIONS = [
  'What is the attention mechanism in transformers?',
  'How does Python\'s GIL affect multithreading?',
  'What is the difference between BERT and GPT?',
  'Explain residual connections and why they help.',
  'What is the walrus operator in Python?',
]

function SkeletonMessage() {
  return (
    <div className="chat-message assistant">
      <div className="message-avatar ai-avatar"><Bot size={16} /></div>
      <div className="message-bubble ai-bubble" style={{ flex: 1 }}>
        <div className="skeleton" style={{ height: 14, width: '80%', marginBottom: 10 }} />
        <div className="skeleton" style={{ height: 14, width: '65%', marginBottom: 10 }} />
        <div className="skeleton" style={{ height: 14, width: '50%' }} />
      </div>
    </div>
  )
}

export default function ChatInterface() {
  const [messages, setMessages]       = useState([])
  const [input, setInput]             = useState('')
  const [loading, setLoading]         = useState(false)
  const [detectHalluc, setDetectHalluc] = useState(true)
  const bottomRef = useRef()
  const textareaRef = useRef()

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const sendMessage = async (query) => {
    const q = (query || input).trim()
    if (!q || loading) return

    const userMsg = { role: 'user', text: q }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setLoading(true)

    try {
      const { data } = await axios.post('/query/', {
        query: q,
        detect_hallucinations: detectHalluc,
      })
      setMessages(prev => [...prev, {
        role: 'assistant',
        text: data.answer,
        citations: data.citations,
        hallucination_flags: data.hallucination_flags,
        confidence: data.confidence,
        latency_ms: data.latency_ms,
      }])
    } catch (err) {
      const detail = err.response?.data?.detail || 'Request failed. Is the backend running?'
      setMessages(prev => [...prev, {
        role: 'assistant',
        text: `⚠️ **Error:** ${detail}`,
        citations: [],
        hallucination_flags: [],
        confidence: null,
        latency_ms: null,
      }])
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage() }
  }

  // Auto-resize textarea
  const handleInput = (e) => {
    setInput(e.target.value)
    const ta = textareaRef.current
    if (ta) { ta.style.height = 'auto'; ta.style.height = `${Math.min(ta.scrollHeight, 150)}px` }
  }

  return (
    <>
      {/* Messages */}
      {messages.length === 0 && !loading ? (
        <div className="empty-state">
          <div className="empty-icon"><MessageSquare size={32} /></div>
          <h2>Ask <span className="gradient-text">My Docs</span></h2>
          <p>Upload documents in the sidebar, then ask anything. Every answer is grounded in your documents with inline citations.</p>
          <div style={{
            display: 'flex',
            alignItems: 'flex-start',
            gap: '10px',
            background: 'rgba(255, 180, 0, 0.08)',
            border: '1px solid rgba(255, 180, 0, 0.25)',
            borderRadius: '12px',
            padding: '12px 16px',
            marginBottom: '20px',
            maxWidth: '540px',
            textAlign: 'left',
          }}>
            <span style={{ fontSize: '1.4rem', lineHeight: 1 }}>💸</span>
            <div>
              <strong style={{ color: '#f59e0b', fontSize: '0.85rem', display: 'block', marginBottom: '3px' }}>
                Heads up — this might be slow!
              </strong>
              <span style={{ color: 'var(--text-muted)', fontSize: '0.80rem', lineHeight: 1.5 }}>
                I am extremely stingy and refuse to pay for API keys, so we're running on free-tier rate limits.
                Responses may take a while. Grab a coffee ☕
              </span>
            </div>
          </div>
          <div className="suggestion-chips">
            {SUGGESTIONS.map(s => (
              <button key={s} className="suggestion-chip" onClick={() => sendMessage(s)}>{s}</button>
            ))}
          </div>
        </div>
      ) : (
        <div className="chat-messages" id="chat-messages">
          {messages.map((msg, i) => (
            <div key={i} className={`chat-message ${msg.role}`} id={`message-${i}`}>
              <div className={`message-avatar ${msg.role === 'user' ? 'user-avatar' : 'ai-avatar'}`}>
                {msg.role === 'user' ? <User size={15} /> : <Bot size={15} />}
              </div>
              <div className={`message-bubble ${msg.role === 'user' ? 'user-bubble' : 'ai-bubble'}`}>
                {msg.role === 'assistant' ? (
                  <>
                    <CitationHighlight text={msg.text} citations={msg.citations} />
                    <CitationList citations={msg.citations} />
                    <HallucinationBadge flags={msg.hallucination_flags} confidence={msg.confidence} />
                    {msg.latency_ms != null && (
                      <div className="message-meta">
                        <span className="latency-tag">{Math.round(msg.latency_ms)}ms</span>
                      </div>
                    )}
                  </>
                ) : (
                  <ReactMarkdown>{msg.text}</ReactMarkdown>
                )}
              </div>
            </div>
          ))}
          {loading && <SkeletonMessage />}
          <div ref={bottomRef} />
        </div>
      )}

      {/* Input */}
      <div className="chat-input-area">
        <div className="chat-input-row">
          <textarea
            id="chat-input"
            ref={textareaRef}
            value={input}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about your documents… (Enter to send, Shift+Enter for newline)"
            rows={1}
            disabled={loading}
          />
          <button
            id="send-button"
            className="send-btn"
            onClick={() => sendMessage()}
            disabled={!input.trim() || loading}
            aria-label="Send message"
          >
            {loading ? <div className="spinner" style={{ width: 18, height: 18 }} /> : <Send size={18} />}
          </button>
        </div>
        <div className="chat-options">
          <label htmlFor="detect-hallucinations">
            <input
              id="detect-hallucinations"
              type="checkbox"
              checked={detectHalluc}
              onChange={e => setDetectHalluc(e.target.checked)}
            />
            Hallucination detection
          </label>
          {messages.length > 0 && (
            <button
              className="btn btn-ghost"
              style={{ fontSize: '0.75rem', padding: '4px 10px', marginLeft: 'auto' }}
              onClick={() => setMessages([])}
              id="clear-chat"
            >
              Clear chat
            </button>
          )}
        </div>
      </div>
    </>
  )
}

import React, { useState, useEffect, useRef } from 'react'
import { sendChat } from '../api/client'
import GlassCard from '../components/GlassCard'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

const SUGGESTIONS = [
  { text: 'How many entities are in pre-transition state?', label: 'Pre-Transitions' },
  { text: 'What is the current entropy trend?', label: 'Entropy Trend' },
  { text: 'Explain the AXIOM-Φ detection system', label: 'AXIOM-Φ System' },
  { text: 'What interventions does ZOLA recommend?', label: 'ZOLA Interventions' },
]

export default function AIAssistant() {
  const [messages, setMessages] = useState(() => {
    const saved = localStorage.getItem('sera_cortex_chat_history')
    if (saved) {
      try {
        return JSON.parse(saved)
      } catch (e) {
        console.error('Failed to parse saved chat history:', e)
      }
    }
    return [
      {
        role: 'ai',
        text: 'SYSTEM ONLINE. Hello! I am SERA-AI, your platform intelligence interface. Ask me anything about resolved entities, entropy status, z-score spikes, or ZOLA causal predictions.'
      }
    ]
  })
  
  const [queryHistory, setQueryHistory] = useState(() => {
    const saved = localStorage.getItem('sera_cortex_query_history')
    if (saved) {
      try {
        return JSON.parse(saved)
      } catch (e) {
        console.error('Failed to parse saved query history:', e)
      }
    }
    return []
  })

  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const chatBottomRef = useRef(null)

  // Save chat history to localStorage
  useEffect(() => {
    localStorage.setItem('sera_cortex_chat_history', JSON.stringify(messages))
  }, [messages])

  // Save query history to localStorage
  useEffect(() => {
    localStorage.setItem('sera_cortex_query_history', JSON.stringify(queryHistory))
  }, [queryHistory])

  // Scroll to bottom on new messages
  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const send = async (text) => {
    const msg = text || input
    if (!msg.trim()) return

    setMessages(prev => [...prev, { role: 'user', text: msg }])
    
    // Add to history list if it's not already there
    setQueryHistory(prev => {
      const filtered = prev.filter(q => q !== msg)
      return [msg, ...filtered].slice(0, 15) // Keep last 15 queries
    })

    setInput('')
    setLoading(true)
    
    try {
      const res = await sendChat(msg)
      setMessages(prev => [...prev, { role: 'ai', text: res.response }])
    } catch {
      setMessages(prev => [...prev, { role: 'ai', text: 'CONNECTION TIMEOUT. Is the platform backend server operational?' }])
    }
    setLoading(false)
  }

  const clearChat = () => {
    setMessages([
      {
        role: 'ai',
        text: 'SYSTEM ONLINE. Hello! I am SERA-AI, your platform intelligence interface. Ask me anything about resolved entities, entropy status, z-score spikes, or ZOLA causal predictions.'
      }
    ])
  }

  const clearHistory = () => {
    setQueryHistory([])
  }

  return (
    <div style={{ animation: 'fadeUp 0.4s ease', display: 'flex', gap: '24px', flexWrap: 'wrap', alignItems: 'stretch' }}>
      <div style={{ flex: '3 1 350px', minWidth: '350px' }}>
        <GlassCard title="Tactical Command Chat Console" glowType="cyan" style={{ padding: 0 }}>
        {/* Suggestion command chips */}
        <div 
          style={{ 
            padding: '16px 20px', 
            borderBottom: '1px solid var(--border)', 
            display: 'flex', 
            gap: 10, 
            flexWrap: 'wrap',
            background: 'rgba(4,7,18,0.2)',
            alignItems: 'center',
            justifyContent: 'space-between'
          }}
        >
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', flex: 1 }}>
            {SUGGESTIONS.map(s => (
              <button 
                key={s.label} 
                className="btn mono" 
                onClick={() => send(s.text)}
                style={{ 
                  background: 'rgba(255,255,255,0.01)', 
                  color: 'var(--text-secondary)', 
                  border: '1px solid var(--border)', 
                  fontSize: 11, 
                  padding: '6px 14px',
                  borderRadius: 20
                }}
              >
                <span>✦</span> {s.label}
              </button>
            ))}
          </div>
          
          <button 
            className="btn mono" 
            onClick={clearChat}
            style={{
              background: 'rgba(255, 75, 75, 0.05)',
              color: '#FF4B4B',
              border: '1px solid rgba(255, 75, 75, 0.2)',
              fontSize: 10,
              padding: '6px 12px',
              borderRadius: 4,
              cursor: 'pointer'
            }}
          >
            CLEAR CONSOLE
          </button>
        </div>
        
        {/* Messages viewport */}
        <div 
          className="chat-messages" 
          style={{ 
            height: 'calc(100vh - 350px)', 
            overflowY: 'auto',
            padding: '24px 28px',
            background: 'radial-gradient(circle at 90% 10%, rgba(58, 134, 255, 0.015) 0%, transparent 60%)'
          }}
        >
          {messages.map((m, i) => (
            <div 
              className={`message ${m.role}`} 
              key={i}
              style={{
                display: 'flex',
                flexDirection: m.role === 'user' ? 'row-reverse' : 'row',
                gap: 12,
                marginBottom: 18,
                alignItems: 'flex-start'
              }}
            >
              {/* Avatar Tag */}
              <div 
                className="mono"
                style={{
                  fontSize: 10,
                  fontWeight: 'bold',
                  padding: '4px 8px',
                  borderRadius: 4,
                  background: m.role === 'user' ? 'rgba(58,134,255,0.15)' : 'rgba(0,245,212,0.15)',
                  color: m.role === 'user' ? 'var(--blue)' : 'var(--cyan)',
                  border: `1px solid ${m.role === 'user' ? 'rgba(58,134,255,0.25)' : 'rgba(0,245,212,0.25)'}`
                }}
              >
                {m.role === 'user' ? 'OPERATOR' : 'SERA-AI'}
              </div>

              {/* Message Bubble */}
              <div 
                className="message-bubble"
                style={{
                  background: m.role === 'user' ? 'rgba(58,134,255,0.06)' : 'rgba(255,255,255,0.015)',
                  border: `1px solid ${m.role === 'user' ? 'rgba(58,134,255,0.15)' : 'var(--border)'}`,
                  color: 'var(--text-primary)',
                  borderRadius: 8,
                  padding: '12px 18px',
                  maxWidth: '75%',
                  fontSize: 13.5,
                  lineHeight: '1.6'
                }}
              >
                {m.role === 'user' ? (
                  m.text
                ) : (
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                      table: ({node, ...props}) => (
                        <div style={{ overflowX: 'auto', margin: '12px 0' }}>
                          <table style={{ width: '100%', borderCollapse: 'collapse', border: '1px solid var(--border)' }} {...props} />
                        </div>
                      ),
                      th: ({node, ...props}) => <th style={{ border: '1px solid var(--border)', padding: '8px 12px', background: 'rgba(0,245,212,0.05)', textAlign: 'left', fontWeight: 'bold', color: 'var(--cyan)', fontSize: 12.5 }} {...props} />,
                      td: ({node, ...props}) => <td style={{ border: '1px solid var(--border)', padding: '8px 12px', fontSize: 12.5 }} {...props} />,
                      code: ({node, inline, ...props}) => inline ? 
                        <code style={{ background: 'rgba(255,255,255,0.08)', padding: '2px 6px', borderRadius: 4, fontFamily: 'monospace', color: 'var(--cyan)', fontSize: 12 }} {...props} /> :
                        <pre style={{ background: 'rgba(0,0,0,0.4)', padding: 12, borderRadius: 6, overflowX: 'auto', border: '1px solid var(--border)', margin: '10px 0' }}><code style={{ fontFamily: 'monospace', color: 'var(--text-primary)', fontSize: 12.5 }} {...props} /></pre>,
                      ul: ({node, ...props}) => <ul style={{ paddingLeft: 20, margin: '8px 0', fontSize: 13 }} {...props} />,
                      ol: ({node, ...props}) => <ol style={{ paddingLeft: 20, margin: '8px 0', fontSize: 13 }} {...props} />,
                      li: ({node, ...props}) => <li style={{ marginBottom: 4 }} {...props} />,
                      p: ({node, ...props}) => <p style={{ margin: '0 0 8px 0' }} {...props} />
                    }}
                  >
                    {m.text}
                  </ReactMarkdown>
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div 
              className="message ai"
              style={{
                display: 'flex',
                gap: 12,
                marginBottom: 18,
                alignItems: 'center'
              }}
            >
              <div 
                className="mono"
                style={{
                  fontSize: 10,
                  fontWeight: 'bold',
                  padding: '4px 8px',
                  borderRadius: 4,
                  background: 'rgba(0,245,212,0.08)',
                  color: 'var(--cyan)',
                  border: '1px solid rgba(0,245,212,0.15)',
                  animation: 'pulse 1.5s infinite'
                }}
              >
                SERA-AI
              </div>
              <div className="mono" style={{ color: 'var(--text-muted)', fontSize: 12 }}>
                Computing response matrix...
              </div>
            </div>
          )}
          <div ref={chatBottomRef} />
        </div>
        
        {/* Input area */}
        <div 
          className="chat-input-row"
          style={{
            display: 'flex',
            gap: 12,
            padding: '20px 28px',
            borderTop: '1px solid var(--border)',
            background: 'rgba(4,7,18,0.45)'
          }}
        >
          <input
            className="glass-input"
            placeholder="Query node stability indexes, predict state transitions, inspect ZOLA..."
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && send()}
            disabled={loading}
          />
         
          <button 
            className="btn btn-primary" 
            onClick={() => send()} 
            disabled={loading || !input.trim()}
            style={{ padding: '0 28px', height: 44 }}
          >
            EXECUTE
          </button>
        </div>
        </GlassCard>
      </div>

      {/* Cortex Active History Log */}
      <div style={{ flex: '1 1 300px', minWidth: '300px', display: 'flex', flexDirection: 'column' }}>
        <GlassCard title="Cortex Active History Log" glowType="cyan">
          {queryHistory.length > 0 && (
            <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 12 }}>
              <button 
                className="btn mono" 
                onClick={clearHistory}
                style={{
                  background: 'rgba(255, 255, 255, 0.02)',
                  color: 'var(--text-secondary)',
                  border: '1px solid var(--border)',
                  fontSize: 10,
                  padding: '4px 10px',
                  borderRadius: 4,
                  cursor: 'pointer'
                }}
              >
                CLEAR LOG
              </button>
            </div>
          )}
          
          {queryHistory.length === 0 ? (
            <div style={{ padding: '20px 10px', textAlign: 'center', color: 'var(--text-muted)', fontSize: 13, fontFamily: 'monospace' }}>
              NO ACTIVE LOGS. EXECUTE QUERIES TO START TRACKING.
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10, maxHeight: 'calc(100vh - 280px)', overflowY: 'auto' }}>
              {queryHistory.map((q, idx) => (
                <div 
                  key={idx}
                  onClick={() => !loading && send(q)}
                  style={{
                    padding: '12px 14px',
                    borderRadius: 6,
                    background: 'rgba(255, 255, 255, 0.01)',
                    border: '1px solid var(--border)',
                    fontSize: 12.5,
                    fontFamily: 'monospace',
                    cursor: loading ? 'not-allowed' : 'pointer',
                    color: 'var(--text-secondary)',
                    transition: 'all 0.2s ease',
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                  }}
                  className="history-item"
                  title={q}
                >
                  <span style={{ color: 'var(--cyan)', marginRight: 6 }}>&gt;</span> {q}
                </div>
              ))}
            </div>
          )}
        </GlassCard>
      </div>
    </div>
  )
}
import { useState } from 'react'
import { sendChat } from '../api/client'
import GlassCard from '../components/GlassCard'
import NewsPanel from '../components/NewsPanel'

const SUGGESTIONS = [
  { text: 'How many entities are in pre-transition state?', label: 'Pre-Transitions' },
  { text: 'What is the current entropy trend?', label: 'Entropy Trend' },
  { text: 'Explain the AXIOM-Φ detection system', label: 'AXIOM-Φ System' },
  { text: 'What interventions does ZOLA recommend?', label: 'ZOLA Interventions' },
]

export default function AIAssistant() {
  const [messages, setMessages] = useState([
    {
      role: 'ai',
      text: 'SYSTEM ONLINE. Hello! I am SERA-AI, your platform intelligence interface. Ask me anything about resolved entities, entropy status, z-score spikes, or ZOLA causal predictions.'
    }
  ])
  
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)

  const send = async (text) => {
    const msg = text || input
    if (!msg.trim()) return

    setMessages(prev => [...prev, { role: 'user', text: msg }])
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
            background: 'rgba(4,7,18,0.2)'
          }}
        >
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
                {m.text}
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
      <div style={{ flex: '1 1 300px', minWidth: '300px', display: 'flex', flexDirection: 'column' }}>
        <NewsPanel domain="all" title="Command Briefing News" />
      </div>
    </div>
  )
}
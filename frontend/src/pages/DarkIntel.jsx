import { useEffect, useState, useRef, useCallback } from 'react'
import GlassCard from '../components/GlassCard'

const BASE = import.meta.env.VITE_API_BASE ?? ''
const AUTH_HEADERS = { 'X-API-Key': import.meta.env.VITE_API_KEY ?? 'sera-demo-2026' }

async function fetchBriefings(clearance = 'ALL') {
  try {
    const url = clearance === 'ALL'
      ? `${BASE}/api/dark-intel/briefings`
      : `${BASE}/api/dark-intel/briefings?clearance=${encodeURIComponent(clearance)}`
    const r = await fetch(url, { headers: AUTH_HEADERS })
    if (!r.ok) return []
    const data = await r.json()
    return Array.isArray(data) ? data : []
  } catch (e) {
    console.error('fetchBriefings failed:', e)
    return []
  }
}

// Decrypt-on-hover text scrambling component
function DecryptText({ text }) {
  const [display, setDisplay] = useState('█'.repeat(Math.max(6, text.length)))
  const [revealed, setRevealed] = useState(false)
  const glyphs = 'X#$_&@?%+=*!₿'
  const timerRef = useRef(null)

  useEffect(() => {
    if (!revealed) {
      setDisplay('█'.repeat(Math.max(6, text.length)))
      return
    }

    let iteration = 0
    clearInterval(timerRef.current)
    
    timerRef.current = setInterval(() => {
      setDisplay(() => {
        return text
          .split('')
          .map((char, index) => {
            if (char === ' ') return ' '
            if (index < iteration) {
              return char
            }
            return glyphs[Math.floor(Math.random() * glyphs.length)]
          })
          .join('')
      })

      iteration += 0.5
      if (iteration >= text.length) {
        clearInterval(timerRef.current)
        setDisplay(text)
      }
    }, 25)

    return () => clearInterval(timerRef.current)
  }, [revealed, text])

  return (
    <span
      className={`redacted ${revealed ? 'revealed' : ''}`}
      onMouseEnter={() => setRevealed(true)}
      onMouseLeave={() => setRevealed(false)}
      style={{ padding: '0 4px', cursor: 'help' }}
    >
      {display}
    </span>
  )
}

// Countdown timer helper component
function ExpiryTimer({ initialSeconds, onExpire }) {
  const [seconds, setSeconds] = useState(initialSeconds)

  useEffect(() => {
    if (seconds <= 0) {
      onExpire()
      return
    }

    const interval = setInterval(() => {
      setSeconds(prev => prev - 1)
    }, 1000)

    return () => clearInterval(interval)
  }, [seconds, onExpire])

  if (seconds <= 0) {
    return <span style={{ color: 'var(--red)', fontWeight: 'bold' }}>DATA WIPED</span>
  }

  const mins = Math.floor(seconds / 60)
  const secs = seconds % 60
  const timeStr = `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`

  return (
    <span className="mono" style={{ color: seconds < 60 ? 'var(--red)' : 'var(--amber)' }}>
      DESTRUCT IN: {timeStr}
    </span>
  )
}

export default function DarkIntel() {
  const [briefs, setBriefs] = useState([])
  const [loading, setLoading] = useState(true)
  const [expiredIds, setExpiredIds] = useState(new Set())
  const [clearanceFilter, setClearanceFilter] = useState('ALL')

  const loadBriefings = useCallback(async (clearance) => {
    setLoading(true)
    const data = await fetchBriefings(clearance)
    setBriefs(data)
    setLoading(false)
  }, [])

  useEffect(() => {
    loadBriefings('ALL')
  }, [])

  const handleClearanceChange = (lvl) => {
    setClearanceFilter(lvl)
    // Pass level or ALL to API
    const apiClearance = lvl === 'ALL' ? 'ALL' : lvl
    loadBriefings(apiClearance)
  }

  const handleExpire = (id) => {
    setExpiredIds(prev => {
      const next = new Set(prev)
      next.add(id)
      return next
    })
  }


  // Parse redacted text and replace [REDACTED] markers with interactive DecryptText components
  const renderRedactedContent = (rawText) => {
    if (!rawText) return null
    
    // Split by bracket markers like [REDACTED] or specific text blocks
    const parts = rawText.split(/\[REDACTED\]/)
    return (
      <>
        {parts.map((part, index) => (
          <span key={index}>
            {part}
            {index < parts.length - 1 && <DecryptText text="CLASSIFIED INFO" />}
          </span>
        ))}
      </>
    )
  }

  const clearanceLevels = ['ALL', 'LEVEL 2 (OPERATOR)', 'LEVEL 3 (ANALYST)', 'LEVEL 4 (DIRECTOR)', 'LEVEL 5 (ADMIN)']

  // Filtering is handled by the API - briefs array already filtered by clearance

  return (
    <div style={{ animation: 'fadeUp 0.4s ease' }} className="crt-overlay">
      <div className="crt-scanline" />
      
      {/* Top Banner Alert */}
      <div 
        className="glass-panel" 
        style={{ 
          padding: '16px 24px', 
          marginBottom: '28px', 
          background: 'rgba(255, 0, 60, 0.05)', 
          border: '1px solid rgba(255, 0, 60, 0.2)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <span className="status-dot" style={{ background: 'var(--red)', boxShadow: '0 0 10px var(--red)', animation: 'pulse 1s infinite' }} />
          <div>
            <h2 className="mono" style={{ fontSize: '15px', fontWeight: '800', color: 'var(--red)', letterSpacing: '2px' }}>
              CLASSIFIED INTELLIGENCE MONITOR
            </h2>
            <p style={{ fontSize: '11px', color: 'var(--text-secondary)', marginTop: '2px' }}>
              RESTRICTED DIRECTIVE 24-B // ACCESS LOGGED BY KRONOS SUB-ROUTINE
            </p>
          </div>
        </div>
        <span 
          className="mono" 
          style={{ 
            background: 'var(--red-dim)', 
            color: 'var(--red)', 
            border: '1px solid var(--red)',
            padding: '4px 10px',
            borderRadius: '4px',
            fontSize: '10px',
            fontWeight: 'bold',
            letterSpacing: '1px'
          }}
        >
          EYES ONLY
        </span>
      </div>

      {/* Clearance Level Filters */}
      <div 
        className="glass-panel" 
        style={{ 
          padding: '16px 20px', 
          marginBottom: '24px', 
          display: 'flex', 
          gap: '12px', 
          alignItems: 'center',
          flexWrap: 'wrap'
        }}
      >
        <span className="mono" style={{ fontSize: '11px', color: 'var(--text-muted)', marginRight: '8px' }}>
          CLEARANCE SELECTOR:
        </span>
        {clearanceLevels.map(lvl => (
          <button
            key={lvl}
            className={`btn mono ${clearanceFilter === lvl ? 'btn-primary' : ''}`}
            onClick={() => handleClearanceChange(lvl)}
            style={{ 
              fontSize: '10px', 
              padding: '6px 12px', 
              textTransform: 'uppercase',
              background: clearanceFilter === lvl ? 'var(--red)' : 'rgba(255,255,255,0.02)',
              color: clearanceFilter === lvl ? '#040712' : 'var(--text-secondary)',
              border: clearanceFilter === lvl ? 'none' : '1px solid var(--border)',
              boxShadow: clearanceFilter === lvl ? '0 0 10px rgba(255,0,60,0.3)' : 'none'
            }}
          >
            {lvl.replace('LEVEL ', 'L-')}
          </button>
        ))}
      </div>

      {/* Intelligence Briefings List */}
      {loading ? (
        <div style={{ padding: '80px', textAlign: 'center', color: 'var(--text-muted)' }} className="mono">
          Decrypting secure databanks...
        </div>
      ) : briefs.length === 0 ? (
        <div style={{ padding: '80px', textAlign: 'center', color: 'var(--text-muted)' }} className="mono">
          No briefings matching active clearance certificate.
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          {briefs.map(brief => {
            const isExpired = expiredIds.has(brief.id)
            const classColor = 
              brief.classification === 'EYES ONLY' ? 'var(--red)' :
              brief.classification === 'TOP SECRET' ? 'var(--magenta)' :
              brief.classification === 'SECRET' ? 'var(--amber)' :
              'var(--cyan)'
            
            return (
              <GlassCard 
                key={brief.id}
                glowType={brief.classification === 'EYES ONLY' || brief.classification === 'TOP SECRET' ? 'red' : 'amber'}
                style={{ 
                  padding: '24px 30px', 
                  borderLeft: `4px solid ${classColor}`,
                  position: 'relative'
                }}
              >
                {/* Header info */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '12px', borderBottom: '1px solid var(--border)', paddingBottom: '14px', marginBottom: '16px' }}>
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <span 
                        className="mono" 
                        style={{ 
                          fontSize: '11px', 
                          fontWeight: '800', 
                          color: classColor,
                          border: `1px solid ${classColor}30`,
                          padding: '2px 8px',
                          borderRadius: '4px',
                          background: `${classColor}08`
                        }}
                      >
                        {brief.classification}
                      </span>
                      <span className="mono" style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                        SOURCE: {brief.source}
                      </span>
                    </div>
                    <h3 className="mono" style={{ fontSize: '18px', fontWeight: '800', marginTop: '8px', color: isExpired ? 'var(--text-muted)' : 'var(--text-primary)' }}>
                      {brief.title}
                    </h3>
                  </div>

                  <div style={{ textAlign: 'right' }} className="mono">
                    <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>ACQUISITION: {brief.date}</div>
                    <div style={{ fontSize: '12px', marginTop: '4px' }}>
                      <ExpiryTimer initialSeconds={brief.expires_in} onExpire={() => handleExpire(brief.id)} />
                    </div>
                  </div>
                </div>

                {/* Brief Content */}
                {isExpired ? (
                  <div 
                    className="mono" 
                    style={{ 
                      background: 'rgba(255,0,60,0.05)', 
                      border: '1px dashed var(--red)', 
                      padding: '20px', 
                      borderRadius: '8px', 
                      color: 'var(--red)', 
                      textAlign: 'center',
                      fontSize: '13px',
                      letterSpacing: '1px'
                    }}
                  >
                    ⚠ SECURITY THREAT EXPOSURE DEFLECTED: THIS SECURE BRIEF HAS SELF-DESTRUCTED AND RE-ENCRYPTED.
                  </div>
                ) : (
                  <div>
                    <p style={{ fontSize: '13.5px', color: 'var(--text-secondary)', lineHeight: '1.6', marginBottom: '18px' }}>
                      {brief.summary}
                    </p>
                    
                    {/* Raw data dump with redaction hover */}
                    <div 
                      style={{ 
                        background: 'rgba(4, 7, 18, 0.6)', 
                        border: '1px solid var(--border)', 
                        borderRadius: '8px', 
                        padding: '16px 20px',
                        position: 'relative'
                      }}
                    >
                      <div className="mono" style={{ fontSize: '9px', color: 'var(--text-muted)', marginBottom: '10px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                        RAW TELEMETRY DECRYPTION MATRIX (HOVER BLACK BLOCKS TO RUN CRACKER)
                      </div>
                      <p className="mono" style={{ fontSize: '12.5px', color: '#4bf5a0', lineHeight: '1.8', textShadow: '0 0 2px rgba(75, 245, 160, 0.2)' }}>
                        {renderRedactedContent(brief.redacted_content)}
                      </p>
                    </div>
                  </div>
                )}

                {/* Card Footer badges */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '16px', fontSize: '10px', color: 'var(--text-muted)' }} className="mono">
                  <span>CLEARANCE REQUIRED: {brief.clearance_level}</span>
                  <span>ID: {brief.id}</span>
                </div>
              </GlassCard>
            )
          })}
        </div>
      )}
    </div>
  )
}

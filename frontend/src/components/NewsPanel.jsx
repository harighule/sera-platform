import { useEffect, useState, useRef } from 'react'
import { fetchNews } from '../api/client'
import GlassCard from './GlassCard'

const SEVERITY_COLORS = {
  critical: 'var(--red)',
  high: 'var(--magenta)',
  medium: 'var(--amber)',
  low: 'var(--cyan)'
}

const SEVERITY_GLOWS = {
  critical: 'glowing-red',
  high: 'glowing-magenta',
  medium: 'glowing-amber',
  low: 'glowing-cyan'
}

const DOMAIN_EMBLEMS = {
  financial: '💳',
  healthcare: '🏥',
  iot: '🔌',
  social: '👥'
}

export default function NewsPanel({ domain = '', title = 'Telemetry Feed' }) {
  const [news, setNews] = useState([])
  const [loading, setLoading] = useState(true)
  const [isGlitching, setIsGlitching] = useState(false)
  const [coords, setCoords] = useState({ lat: 34.0522, lng: -118.2437 })

  // Coordinates drift
  useEffect(() => {
    const id = setInterval(() => {
      setCoords(prev => ({
        lat: prev.lat + (Math.random() - 0.5) * 0.0006,
        lng: prev.lng + (Math.random() - 0.5) * 0.0006,
      }))
    }, 3000)
    return () => clearInterval(id)
  }, [])

  useEffect(() => {
    let active = true
    setIsGlitching(true)
    setLoading(true)

    const glitchTimer = setTimeout(() => {
      if (active) setIsGlitching(false)
    }, 450)

    const loadData = () => {
      fetchNews(domain).then(data => {
        if (active) {
          setNews(data)
          setLoading(false)
        }
      })
    }

    loadData()
    const interval = setInterval(loadData, 12000) // Poll news every 12 seconds

    return () => {
      active = false
      clearTimeout(glitchTimer)
      clearInterval(interval)
    }
  }, [domain])

  return (
    <div className="glass-panel" style={{ padding: '20px', height: '100%', display: 'flex', flexDirection: 'column', minHeight: '420px', position: 'relative', overflow: 'hidden' }}>
      {/* Raster Scanline Overlay */}
      <div className="tv-raster-overlay" style={{ borderRadius: '16px' }} />

      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px', borderBottom: '1px solid var(--border)', paddingBottom: '10px', zIndex: 12 }}>
        <h3 className="mono" style={{ fontSize: '12px', fontWeight: '800', letterSpacing: '1.5px', textTransform: 'uppercase', color: 'var(--cyan)', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ display: 'inline-block', width: '6px', height: '6px', borderRadius: '50%', background: 'var(--red)', boxShadow: '0 0 6px var(--red)' }} className="status-dot" />
          {title}
        </h3>
        <span className="mono" style={{ fontSize: '8px', color: 'var(--text-muted)' }}>
          LOC: {coords.lat.toFixed(4)}N / {Math.abs(coords.lng).toFixed(4)}W
        </span>
      </div>

      {/* Main Area */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', relative: true, minHeight: 0, zIndex: 12 }}>
        {isGlitching ? (
          /* Glitch/Static screen */
          <div className="tv-static-screen" style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '8px', borderRadius: '10px', border: '1px solid var(--border)' }}>
            <span className="mono text-glow" style={{ fontSize: '10px', color: 'var(--red)', fontWeight: '900', letterSpacing: '2px', animation: 'pulse 1s infinite' }}>
              CONNECTING...
            </span>
            <span className="mono" style={{ fontSize: '8px', color: 'var(--text-muted)' }}>Tuning receiver signal feed...</span>
          </div>
        ) : loading && news.length === 0 ? (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', flex: 1, color: 'var(--text-muted)' }} className="mono">
            Connecting to SIGINT feeds...
          </div>
        ) : news.length === 0 ? (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', flex: 1, color: 'var(--text-muted)' }} className="mono">
            No intelligence warnings registered.
          </div>
        ) : (
          <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '12px', paddingRight: '4px' }}>
            {news.map((item) => {
              const sevColor = SEVERITY_COLORS[item.severity] || 'var(--text-muted)'
              const domainEmblem = DOMAIN_EMBLEMS[item.domain] || '◈'
              
              return (
                <div 
                  key={item.id} 
                  className="news-card"
                  style={{ 
                    background: 'rgba(5, 9, 25, 0.45)', 
                    border: '1px solid var(--border)', 
                    borderRadius: '10px', 
                    padding: '12px', 
                    transition: 'all 0.25s ease',
                    position: 'relative',
                    overflow: 'hidden'
                  }}
                >
                  {/* Accent Top Border Indicator */}
                  <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: '2px', background: sevColor, opacity: 0.8 }} />

                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px', marginTop: '2px' }}>
                    <span className="mono" style={{ fontSize: '9px', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                      <span style={{ fontSize: '11px' }}>{domainEmblem}</span>
                      {item.source}
                    </span>
                    
                    {/* Severity Badge */}
                    <span 
                      className="mono" 
                      style={{ 
                        fontSize: '8px', 
                        fontWeight: '700', 
                        color: sevColor, 
                        border: `1px solid ${sevColor}40`, 
                        background: `${sevColor}10`,
                        padding: '2px 6px',
                        borderRadius: '4px',
                        textTransform: 'uppercase',
                        letterSpacing: '0.5px'
                      }}
                    >
                      {item.severity}
                    </span>
                  </div>

                  <h4 style={{ fontSize: '13px', fontWeight: '700', color: 'var(--text-primary)', marginBottom: '6px', lineHeight: '1.3' }}>
                    {item.title}
                  </h4>
                  
                  <p style={{ fontSize: '11px', color: 'var(--text-secondary)', lineHeight: '1.4', marginBottom: '8px' }}>
                    {item.summary}
                  </p>

                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '9px', color: 'var(--text-muted)' }} className="mono">
                    <span>{item.timestamp}</span>
                    <div style={{ display: 'flex', gap: '4px' }}>
                      {item.tags?.map((tag, idx) => (
                        <span key={idx} style={{ background: 'rgba(255,255,255,0.03)', padding: '1px 4px', borderRadius: '3px' }}>
                          #{tag}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>

      {/* Audio Visualizer strip */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '12px', borderTop: '1px solid var(--border)', paddingTop: '8px', fontSize: '9px', color: 'var(--text-muted)', zIndex: 12 }} className="mono shrink-0">
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span>AUDIO_SIG:</span>
          <div style={{ display: 'flex', alignItems: 'end', gap: '2px', height: '10px' }}>
            <span className="animate-eq-1" style={{ width: '2px', background: 'var(--cyan)' }} />
            <span className="animate-eq-3" style={{ width: '2px', background: 'var(--cyan)' }} />
            <span className="animate-eq-2" style={{ width: '2px', background: 'var(--cyan)' }} />
            <span className="animate-eq-5" style={{ width: '2px', background: 'var(--amber)' }} />
            <span className="animate-eq-4" style={{ width: '2px', background: 'var(--red)' }} />
          </div>
        </div>
        <span>AUT_MON: ON</span>
      </div>

      {/* CNBC/Bloomberg Style Index Ticker */}
      <div style={{ background: '#02040a', border: '1px solid var(--border)', borderRadius: '6px', height: '22px', display: 'flex', alignItems: 'center', overflow: 'hidden', marginTop: '8px', zIndex: 12 }} className="mono shrink-0">
        <div style={{ background: 'var(--red)', color: '#040712', fontSize: '8px', fontWeight: '900', padding: '2px 6px', textTransform: 'uppercase', letterSpacing: '1px', display: 'flex', alignItems: 'center', height: '100%', flexShrink: 0 }}>
          INDEX
        </div>
        <div style={{ flex: 1, overflow: 'hidden', relative: true }}>
          <div className="animate-marquee-scroll" style={{ display: 'flex', gap: '30px', fontSize: '8px', color: 'var(--text-secondary)' }}>
            {[0, 1].map(j => (
              <div key={j} style={{ display: 'flex', gap: '30px', alignItems: 'center', whiteSpace: 'nowrap' }}>
                <span>APEX_INTEGRITY: 99.8% <span style={{ color: 'var(--cyan)' }}>▲</span></span>
                <span>CYBER_ALERT: ELEVATED <span style={{ color: 'var(--red)' }}>●</span></span>
                <span>BTC/USD: $94,850 +2.1% <span style={{ color: 'var(--cyan)' }}>▲</span></span>
                <span>ZULU_TIME: {new Date().toISOString().split('T')[1].substring(0, 5)}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

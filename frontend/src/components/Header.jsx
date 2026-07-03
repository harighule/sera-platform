import { useEffect, useState } from 'react'

export default function Header({ title, subtitle }) {
  const [time, setTime] = useState('')

  useEffect(() => {
    const updateTime = () => {
      const now = new Date()
      setTime(now.toLocaleTimeString() + ' | ' + now.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }))
    }
    updateTime()
    const interval = setInterval(updateTime, 1000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="header">
      <div>
        <div className="header-title">{title}</div>
        {subtitle && <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>{subtitle}</div>}
      </div>
      
      <div className="header-badges">
        <div className="mono" style={{ fontSize: 11, color: 'var(--text-muted)', marginRight: 16, borderRight: '1px solid var(--border)', paddingRight: 16 }}>
          {time}
        </div>
        
        <span className="badge badge-cyan" style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
          <span className="status-dot" style={{ margin: 0, width: 6, height: 6 }} />
          LIVE TELEMETRY
        </span>
        
        <span className="badge badge-amber" style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
          <span>∿</span> AXIOM-Φ ACTIVE
        </span>
        
        <span 
          className="badge mono" 
          style={{ 
            background: 'rgba(131, 56, 236, 0.12)', 
            color: '#b388ff', 
            border: '1px solid rgba(131, 56, 236, 0.25)',
            textShadow: '0 0 10px rgba(131, 56, 236, 0.3)'
          }}
        >
          KRONOS v1.0
        </span>
      </div>
    </div>
  )
}
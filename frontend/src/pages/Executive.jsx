import { useEffect, useState } from 'react'
import { fetchExecutiveMovements } from '../api/client'
import GlassCard from '../components/GlassCard'
import './ExecutiveDashboard.css'

export default function Executive() {
  const [movements, setMovements] = useState([])
  const [last7DaysCount, setLast7DaysCount] = useState(0)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchExecutiveMovements()
      .then(res => {
        if (res) {
          setMovements(res.movements || [])
          setLast7DaysCount(res.last_7_days_count || 0)
        }
      })
      .catch(err => console.error(err))
      .finally(() => setLoading(false))
  }, [])

  const getBadgeStyle = (type) => {
    switch (type) {
      case 'hire':
        return {
          background: 'rgba(0, 245, 212, 0.08)',
          color: 'var(--cyan)',
          border: '1px solid rgba(0, 245, 212, 0.3)',
        }
      case 'departure':
        return {
          background: 'rgba(255, 0, 110, 0.08)',
          color: 'var(--magenta)',
          border: '1px solid rgba(255, 0, 110, 0.3)',
        }
      case 'promotion':
      default:
        return {
          background: 'rgba(58, 134, 255, 0.08)',
          color: 'var(--blue)',
          border: '1px solid rgba(58, 134, 255, 0.3)',
        }
    }
  }

  const formatBadgeText = (type) => {
    switch (type) {
      case 'hire':
        return '🆕 NEW APPOINTMENT'
      case 'departure':
        return '🚫 DEPARTURE / EXIT'
      case 'promotion':
      default:
        return '⇅ PROMOTION / ALIGNMENT'
    }
  }

  const formatDate = (dateStr) => {
    if (!dateStr) return 'N/A'
    try {
      const d = new Date(dateStr)
      return d.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' })
    } catch {
      return dateStr
    }
  }

  if (loading) {
    return <div className="loading-container">Synchronizing Executive movements data...</div>
  }

  return (
    <div style={{ animation: 'fadeUp 0.4s ease', display: 'flex', flexDirection: 'column', gap: '24px' }}>
      
      {/* Top count summary */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '20px' }}>
        <GlassCard title="Recent Transition Events" glowType="blue">
          <div style={{ padding: '15px 0' }}>
            <div className="mono" style={{ fontSize: '32px', color: 'var(--blue)', fontWeight: 'bold' }}>
              {last7DaysCount}
            </div>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '1px', marginTop: '4px' }}>
              Movements in Last 7 Days
            </div>
          </div>
        </GlassCard>

        <GlassCard title="Total Monitored Scope" glowType="cyan">
          <div style={{ padding: '15px 0' }}>
            <div className="mono" style={{ fontSize: '32px', color: 'var(--cyan)', fontWeight: 'bold' }}>
              {movements.length}
            </div>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '1px', marginTop: '4px' }}>
              Recent Tracked Changes (100 Max)
            </div>
          </div>
        </GlassCard>
      </div>

      {/* Main split: Timeline & Structured list */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '24px', alignItems: 'start' }}>
        
        {/* Timeline block */}
        <GlassCard title="Executive Intel Timeline" subtitle="Sequential order of public LinkedIn delta events">
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', marginTop: '20px', paddingLeft: '8px' }}>
            {movements.slice(0, 5).map((m, idx) => (
              <div 
                key={m.id || idx} 
                style={{ 
                  borderLeft: '2px solid var(--border)', 
                  paddingLeft: '16px', 
                  position: 'relative',
                  paddingBottom: '12px' 
                }}
              >
                {/* Circle marker */}
                <div style={{
                  width: '10px',
                  height: '10px',
                  borderRadius: '50%',
                  background: m.change_type === 'hire' ? 'var(--cyan)' : m.change_type === 'departure' ? 'var(--magenta)' : 'var(--blue)',
                  position: 'absolute',
                  left: '-6px',
                  top: '4px',
                  boxShadow: `0 0 6px ${m.change_type === 'hire' ? 'var(--cyan)' : m.change_type === 'departure' ? 'var(--magenta)' : 'var(--blue)'}`
                }} />
                
                <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginBottom: '4px' }} className="mono">
                  {formatDate(m.change_date)}
                </div>
                <div style={{ fontSize: '13px', fontWeight: 'bold', color: 'var(--text-primary)' }}>
                  {m.exec_name}
                </div>
                <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
                  {m.ticker} — {m.new_title || 'Role Transitioned'}
                </div>
              </div>
            ))}
            {movements.length === 0 && (
              <div style={{ color: 'var(--text-muted)', fontSize: '12px' }}>No recent executive changes found.</div>
            )}
          </div>
        </GlassCard>

        {/* Structured movement table */}
        <GlassCard title="Tracked Executive Alignments" subtitle="Detailed records of CEO, CFO, and Board movements">
          <div className="table-container" style={{ marginTop: '16px' }}>
            <table className="glass-table" style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border)', color: 'var(--text-muted)', fontSize: '11px', textTransform: 'uppercase', letterSpacing: '1px' }}>
                  <th style={{ padding: '12px 16px' }}>Company</th>
                  <th style={{ padding: '12px 16px' }}>Executive Name</th>
                  <th style={{ padding: '12px 16px' }}>Prior Title</th>
                  <th style={{ padding: '12px 16px' }}>Current Title</th>
                  <th style={{ padding: '12px 16px' }}>Change Type</th>
                </tr>
              </thead>
              <tbody>
                {movements.map((m, idx) => (
                  <tr key={m.id || idx} style={{ borderBottom: '1px solid rgba(255,255,255,0.02)' }} className="table-row-hover">
                    <td style={{ padding: '12px 16px', fontWeight: 'bold', color: 'var(--text-primary)' }}>
                      {m.ticker}
                    </td>
                    <td style={{ padding: '12px 16px', fontWeight: '500' }}>
                      {m.exec_name}
                    </td>
                    <td style={{ padding: '12px 16px', color: 'var(--text-secondary)', fontSize: '13px' }}>
                      {m.old_title || <span style={{ color: 'var(--text-muted)', fontSize: '11px', fontStyle: 'italic' }}>None (External)</span>}
                    </td>
                    <td style={{ padding: '12px 16px', color: 'var(--text-primary)', fontSize: '13px', fontWeight: '500' }}>
                      {m.new_title || <span style={{ color: 'var(--text-muted)', fontSize: '11px', fontStyle: 'italic' }}>Role Exited</span>}
                    </td>
                    <td style={{ padding: '12px 16px' }}>
                      <span 
                        style={{ 
                          ...getBadgeStyle(m.change_type),
                          padding: '4px 8px',
                          borderRadius: '4px',
                          fontSize: '9px',
                          fontWeight: 'bold',
                          letterSpacing: '0.5px',
                          display: 'inline-block',
                          whiteSpace: 'nowrap'
                        }}
                      >
                        {formatBadgeText(m.change_type)}
                      </span>
                    </td>
                  </tr>
                ))}
                {movements.length === 0 && (
                  <tr>
                    <td colSpan="5" style={{ padding: '24px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '13px' }}>
                      No executive changes detected in registry.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </GlassCard>

      </div>

    </div>
  )
}

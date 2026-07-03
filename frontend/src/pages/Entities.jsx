import { useEffect, useState } from 'react'
import { fetchEntities } from '../api/client'
import GlassCard from '../components/GlassCard'
import NewsPanel from '../components/NewsPanel'

const DOMAIN_ICONS = {
  financial: { icon: '💳', color: '#3a86ff' },
  healthcare: { icon: '🏥', color: '#00f5d4' },
  iot: { icon: '🔌', color: '#ffbe0b' },
  social: { icon: '👥', color: '#ff006e' }
}

export default function Entities() {
  const [entities, setEntities] = useState([])
  const [total, setTotal] = useState(0)
  const [search, setSearch] = useState('')
  const [domainFilter, setDomainFilter] = useState('all')
  const [statusFilter, setStatusFilter] = useState('all')

  useEffect(() => {
    fetchEntities().then(res => {
      if (res) {
        setEntities(res.entities || [])
        setTotal(res.total || 0)
      }
    })
  }, [])

  // Filter logic
  const filtered = entities.filter(e => {
    const matchesSearch = e.name.toLowerCase().includes(search.toLowerCase()) ||
                          e.domain.toLowerCase().includes(search.toLowerCase())
    const matchesDomain = domainFilter === 'all' || e.domain.toLowerCase() === domainFilter
    const matchesStatus = statusFilter === 'all' || e.status === statusFilter
    
    return matchesSearch && matchesDomain && matchesStatus
  })

  // Get status color coding
  const getStatusGlow = (status) => {
    if (status === 'critical') return 'red'
    if (status === 'pre-transition') return 'amber'
    return 'cyan'
  }

  return (
    <div style={{ animation: 'fadeUp 0.4s ease', display: 'flex', gap: '24px', flexWrap: 'wrap', alignItems: 'stretch' }}>
      <div style={{ flex: '3 1 350px', minWidth: '350px' }}>
      {/* Filters Header toolbar */}
      <div 
        className="glass-panel" 
        style={{ 
          padding: 20, 
          marginBottom: 30, 
          display: 'flex', 
          flexWrap: 'wrap', 
          gap: 16, 
          alignItems: 'center',
          justifyContent: 'space-between'
        }}
      >
        <div style={{ flex: 1, minWidth: 260 }}>
          <input
            className="glass-input"
            placeholder="Search entities by name, domain, ID..."
            value={search}
            onChange={e => setSearch(e.target.value)}
          />
        </div>
        
        {/* Domain Filter Buttons */}
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          {['all', 'financial', 'healthcare', 'iot', 'social'].map(dom => (
            <button
              key={dom}
              className={`btn ${domainFilter === dom ? 'btn-primary' : ''}`}
              onClick={() => setDomainFilter(dom)}
              style={{ 
                padding: '6px 12px', 
                fontSize: 12,
                textTransform: 'capitalize',
                background: domainFilter === dom ? undefined : 'rgba(255,255,255,0.02)'
              }}
            >
              {dom === 'all' ? 'All Domains' : dom}
            </button>
          ))}
        </div>

        {/* Status Filter Buttons */}
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', borderLeft: '1px solid var(--border)', paddingLeft: 16 }}>
          {['all', 'stable', 'pre-transition', 'critical'].map(stat => (
            <button
              key={stat}
              className={`btn ${statusFilter === stat ? 'btn-primary' : ''}`}
              onClick={() => setStatusFilter(stat)}
              style={{ 
                padding: '6px 12px', 
                fontSize: 12,
                textTransform: 'capitalize',
                background: statusFilter === stat ? undefined : 'rgba(255,255,255,0.02)'
              }}
            >
              {stat === 'all' ? 'All Statuses' : stat.replace('-', ' ')}
            </button>
          ))}
        </div>
      </div>

      <div style={{ color: 'var(--text-muted)', fontSize: '12px', marginBottom: '12px', paddingLeft: '4px' }} className="mono">
        Showing {entities.length} of {total} entities
      </div>

      {filtered.length === 0 ? (
        <GlassCard style={{ textAlign: 'center', padding: '60px 20px', color: 'var(--text-muted)' }}>
          No matching entities found in the resolution registry.
        </GlassCard>
      ) : (
        <div className="grid-3">
          {filtered.map(e => {
            const domInfo = DOMAIN_ICONS[e.domain.toLowerCase()] || { icon: '◈', color: 'var(--cyan)' }
            const entropyPct = Math.min((e.entropy / 4.0) * 100, 100)
            const r = 24
            const circ = 2 * Math.PI * r
            const strokeDashoffset = circ - (entropyPct / 100) * circ
            
            return (
              <GlassCard 
                key={e.id} 
                glowType={getStatusGlow(e.status)}
                style={{ 
                  display: 'flex', 
                  flexDirection: 'column', 
                  justifyContent: 'space-between',
                  minHeight: 210,
                  position: 'relative'
                }}
              >
                {/* Card Top */}
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
                    <div>
                      <span className="mono" style={{ color: 'var(--text-muted)', fontSize: 10 }}>ID: {e.id}</span>
                      <h3 style={{ fontSize: 17, fontWeight: 700, marginTop: 2, color: 'var(--text-primary)' }}>{e.name}</h3>
                    </div>
                    {/* Domain badge symbol */}
                    <span 
                      style={{ 
                        fontSize: 18, 
                        padding: 8, 
                        background: `rgba(${domInfo.color === '#3a86ff' ? '58,134,255' : domInfo.color === '#00f5d4' ? '0,245,212' : domInfo.color === '#ffbe0b' ? '255,190,11' : '255,0,110'}, 0.08)`, 
                        borderRadius: 8,
                        border: `1px solid rgba(${domInfo.color === '#3a86ff' ? '58,134,255' : domInfo.color === '#00f5d4' ? '0,245,212' : domInfo.color === '#ffbe0b' ? '255,190,11' : '255,0,110'}, 0.2)`
                      }}
                      title={e.domain}
                    >
                      {domInfo.icon}
                    </span>
                  </div>
                  
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 14 }}>
                    <span className={`status status-${e.status.replace('pre-transition', 'pre')}`}>
                      {e.status}
                    </span>
                    <span style={{ color: 'var(--text-muted)', fontSize: 12, textTransform: 'capitalize' }}>
                      {e.domain}
                    </span>
                  </div>
                </div>

                {/* Card Bottom: Entropy Gauge Circle */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderTop: '1px solid rgba(255,255,255,0.03)', paddingTop: 14, marginTop: 12 }}>
                  <div>
                    <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>TELEMETRY COUNT</div>
                    <div className="mono" style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-secondary)' }}>{e.event_count ?? 0} signals</div>
                  </div>
                  
                  {/* Circular SVG Gauge */}
                  <div style={{ position: 'relative', width: 56, height: 56, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <svg width="56" height="56" style={{ transform: 'rotate(-90deg)' }}>
                      <circle cx="28" cy="28" r={r} fill="transparent" stroke="rgba(255,255,255,0.04)" strokeWidth="4" />
                      <circle 
                        cx="28" 
                        cy="28" 
                        r={r} 
                        fill="transparent" 
                        stroke={
                          e.status === 'critical' ? 'var(--red)' : 
                          e.status === 'pre-transition' ? 'var(--amber)' : 
                          'var(--cyan)'
                        } 
                        strokeWidth="4" 
                        strokeDasharray={circ}
                        strokeDashoffset={strokeDashoffset}
                        style={{ transition: 'stroke-dashoffset 0.6s ease' }}
                      />
                    </svg>
                    <div 
                      className="mono" 
                      style={{ 
                        position: 'absolute', 
                        fontSize: 10, 
                        fontWeight: 'bold',
                        color: 
                          e.status === 'critical' ? 'var(--red)' : 
                          e.status === 'pre-transition' ? 'var(--amber)' : 
                          'var(--cyan)'
                      }}
                    >
                      {typeof e.entropy === 'number' ? e.entropy.toFixed(2) : e.entropy}
                    </div>
                  </div>
                </div>
              </GlassCard>
            )
          })}
        </div>
      )}
      </div>
      <div style={{ flex: '1 1 300px', minWidth: '300px', display: 'flex', flexDirection: 'column' }}>
        <NewsPanel domain={domainFilter} title={`${domainFilter === 'all' ? 'System' : domainFilter} Intel Warnings`} />
      </div>
    </div>
  )
}
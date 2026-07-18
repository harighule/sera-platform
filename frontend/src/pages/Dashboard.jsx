import { useEffect, useState, useRef } from 'react'
import { fetchStats, fetchFreshness, fetchEntities, fetchNarrativeExpansion } from '../api/client'
import { createStream } from '../api/websocket'
import AnimatedCounter from '../components/AnimatedCounter'
import GlassCard from '../components/GlassCard'
import { useToast } from '../components/ToastNotification'
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'
import NewsPanel from '../components/NewsPanel'

const PROTO_CLASS = { SWIFT: 'proto-swift', FHIR: 'proto-fhir', MQTT: 'proto-mqtt', HTTP: 'proto-http' }
const PROTO_ICONS = { SWIFT: '💳', FHIR: '🏥', MQTT: '🔌', HTTP: '🌐' }

export default function Dashboard() {
  const [stats, setStats] = useState(null)
  const [feed, setFeed] = useState([])
  const [history, setHistory] = useState([])
  const [freshness, setFreshness] = useState(null)
  const [topExpand, setTopExpand] = useState([])
  const [sectorStats, setSectorStats] = useState([])
  const [narratives, setNarratives] = useState([])
  const prevAlertsRef = useRef(0)
  const wsRef = useRef(null)
  const { addToast } = useToast()

  useEffect(() => {
    // Initial fetch
    fetchStats().then(data => {
      if (data) {
        setStats(data)
        prevAlertsRef.current = data.active_alerts ?? 0
        setHistory([{
          time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
          eps: data.events_per_second ?? 0,
          entropy: data.entropy_average ?? 0
        }])
      }
    })

    // Fetch data freshness and entities
    fetchFreshness().then(data => {
      if (data) setFreshness(data)
    })
    
    fetchEntities().then(res => {
      if (res && res.entities) {
        const sorted = [...res.entities].sort((a, b) => (b.expansion_score ?? 0) - (a.expansion_score ?? 0))
        setTopExpand(sorted.slice(0, 5))
        
        const sectors = {}
        res.entities.forEach(e => {
          const sec = e.domain || 'Other'
          sectors[sec] = (sectors[sec] || 0) + (e.event_count || 0)
        })
        const sectorData = Object.keys(sectors).map(k => ({
          name: k,
          count: sectors[k]
        })).sort((a, b) => b.count - a.count)
        setSectorStats(sectorData)
        
        const tickers = sorted.slice(0, 3).map(e => e.ticker).filter(Boolean)
        Promise.all(tickers.map(t => fetchNarrativeExpansion(t))).then(reports => {
          const valid = reports.filter(r => r && r.summary)
          setNarratives(valid)
        })
      }
    })

    // Poll stats & freshness
    const interval = setInterval(() => {
      fetchStats().then(data => {
        if (data) {
          setStats(data)
          
          if (data.active_alerts > prevAlertsRef.current) {
            addToast(`AXIOM-Φ: ${data.active_alerts - prevAlertsRef.current} new pre-transition behavior alerts detected!`, 'critical')
          }
          prevAlertsRef.current = data.active_alerts ?? 0

          setHistory(prev => [
            ...prev,
            {
              time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
              eps: parseFloat(data.events_per_second) || 0,
              entropy: parseFloat(data.entropy_average) || 0
            }
          ].slice(-15))
        }
      })

      fetchFreshness().then(data => {
        if (data) setFreshness(data)
      })
    }, 5000)

    // Listen to WebSocket Event Stream
    wsRef.current = createStream((data) => {
      if (data.event) {
        const ev = data.event
        setFeed(prev => [ev, ...prev].slice(0, 30))
        
        if (ev.event_type === 'CYBERSPACE_CRAWL_COMPLETE') {
          addToast("KRONOS: Cyberspace learning completed. Parameters updated.", "success")
        }
      }
    })

    return () => {
      clearInterval(interval)
      wsRef.current?.close()
    }
  }, [addToast])

  return (
    <div style={{ animation: 'fadeUp 0.4s ease', display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Freshness Banner Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
        <h1 style={{ margin: 0, fontSize: '28px', fontWeight: 700 }}>Telemetry Command Deck</h1>
        {freshness && (
          <div className="freshness-badge mono" style={{
            padding: '6px 12px',
            borderRadius: '6px',
            fontSize: '11px',
            fontWeight: 600,
            background: freshness.status === 'fresh' ? 'rgba(16, 185, 129, 0.1)' : freshness.status === 'warning' ? 'rgba(245, 158, 11, 0.1)' : 'rgba(239, 68, 68, 0.1)',
            border: `1px solid ${freshness.status === 'fresh' ? '#10b981' : freshness.status === 'warning' ? '#f59e0b' : '#ef4444'}`,
            color: freshness.status === 'fresh' ? '#34d399' : freshness.status === 'warning' ? '#fbbf24' : '#f87171',
            boxShadow: `0 0 10px ${freshness.status === 'fresh' ? 'rgba(16, 185, 129, 0.1)' : freshness.status === 'warning' ? 'rgba(245, 158, 11, 0.1)' : 'rgba(239, 68, 68, 0.1)'}`
          }}>
            ● DATA FRESHNESS: {freshness.status.toUpperCase()} 
            {freshness.alert && <span style={{ marginLeft: '8px', opacity: 0.8 }}>({freshness.alert})</span>}
          </div>
        )}
      </div>

      <div style={{ display: 'flex', gap: '24px', flexWrap: 'wrap', alignItems: 'stretch' }}>
        <div style={{ flex: '3 1 350px', minWidth: '350px' }}>
          <div className="stats-grid">
            {[
              { label: 'Total Entities', value: stats?.total_entities ?? 0, sub: 'Resolved profiles', glow: 'blue' },
              { label: 'Active Alerts', value: stats?.active_alerts ?? 0, sub: 'Pre-transition states', glow: stats?.active_alerts > 0 ? 'red' : '' },
              { label: 'Events / sec', value: stats?.events_per_second ?? 0, sub: 'Live ingestion rate', glow: 'cyan' },
              { label: 'Avg Entropy', value: stats?.entropy_average ?? 0, sub: 'AXIOM-Φ score', glow: stats?.entropy_average > 1.5 ? 'amber' : 'cyan' },
              { label: 'Events Processed', value: stats?.events_processed ?? 0, sub: 'EVENTS PROCESSED', glow: '' },
              { label: 'Active Protocols', value: stats?.protocols_active ?? 0, sub: 'SWIFT · FHIR · MQTT · HTTP', glow: '' },
            ].map(s => (
              <GlassCard key={s.label} glowType={s.glow}>
                <div className="stat-label">{s.label}</div>
                <div className="stat-value mono">
                  <AnimatedCounter value={s.value} />
                </div>
                <div className="stat-sub">{s.sub}</div>
              </GlassCard>
            ))}
          </div>

          {/* visual metrics & narrative rows */}
          <div className="grid-2" style={{ marginBottom: 30 }}>
            {/* Visual Chart Card */}
            <GlassCard title="Telemetric Flow & Threat Potential" glowType="cyan">
              <div style={{ height: 260, minHeight: 260, minWidth: 0, display: 'block', marginTop: 10 }}>
                {history.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={history} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                      <defs>
                        <linearGradient id="colorEps" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="var(--cyan)" stopOpacity={0.3}/>
                          <stop offset="95%" stopColor="var(--cyan)" stopOpacity={0}/>
                        </linearGradient>
                        <linearGradient id="colorEntropy" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="var(--blue)" stopOpacity={0.3}/>
                          <stop offset="95%" stopColor="var(--blue)" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <XAxis dataKey="time" stroke="var(--text-muted)" fontSize={9} tickLine={false} />
                      <YAxis stroke="var(--text-muted)" fontSize={9} tickLine={false} />
                      <Tooltip 
                        contentStyle={{ 
                          background: 'rgba(10, 15, 30, 0.9)', 
                          borderColor: 'var(--border)',
                          borderRadius: 8,
                          fontSize: 12,
                          color: 'var(--text-primary)'
                        }} 
                      />
                      <Area type="monotone" name="Ingestion Rate (EPS)" dataKey="eps" stroke="var(--cyan)" fillOpacity={1} fill="url(#colorEps)" strokeWidth={1.5} />
                      <Area type="monotone" name="Avg Entropy" dataKey="entropy" stroke="var(--blue)" fillOpacity={1} fill="url(#colorEntropy)" strokeWidth={1.5} />
                    </AreaChart>
                  </ResponsiveContainer>
                ) : (
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-muted)', fontSize: 13 }} className="mono">
                    Awaiting first telemetry sync...
                  </div>
                )}
              </div>
            </GlassCard>

            {/* Live Stream Card */}
            <GlassCard title="Live Ingestion Stream Feed" glowType="blue">
              <div className="live-feed">
                {feed.length === 0 && (
                  <div style={{ color: 'var(--text-muted)', fontSize: 13, display: 'flex', alignItems: 'center', gap: 10, height: 200, justifyContent: 'center' }} className="mono">
                    <span className="status-dot" />
                    Listening for multi-protocol signals...
                  </div>
                )}
                {feed.map((ev, i) => (
                  <div className="feed-item" key={i}>
                    <span className={`protocol-badge ${PROTO_CLASS[ev.protocol] || ''}`}>
                      {PROTO_ICONS[ev.protocol]} {ev.protocol}
                    </span>
                    <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{ev.name}</span>
                    <span style={{ color: 'var(--text-muted)' }}>→</span>
                    <span style={{ color: 'var(--text-secondary)' }}>{ev.event_type?.replace(/_/g, ' ')}</span>
                  </div>
                ))}
              </div>
            </GlassCard>
          </div>

          {/* New Commercial Intelligence Row */}
          <div className="grid-2" style={{ marginBottom: 30 }}>
            {/* Top Expansion Candidates */}
            <GlassCard title="Top Expansion Candidates" glowType="amber">
              <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', marginTop: '10px' }}>
                {topExpand.length > 0 ? topExpand.map((company, index) => (
                  <div key={company.id} style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '13px' }}>
                      <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                        {index + 1}. {company.name} ({company.ticker})
                      </span>
                      <span className="mono" style={{ color: '#fbbf24', fontWeight: 600 }}>
                        {((company.expansion_score || 0.5) * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div style={{ width: '100%', height: '6px', background: 'rgba(255,255,255,0.05)', borderRadius: '3px', overflow: 'hidden' }}>
                      <div style={{
                        width: `${(company.expansion_score || 0.5) * 100}%`,
                        height: '100%',
                        background: 'linear-gradient(90deg, #f59e0b, #fbbf24)',
                        borderRadius: '3px',
                        boxShadow: '0 0 8px rgba(245, 158, 11, 0.4)'
                      }} />
                    </div>
                  </div>
                )) : (
                  <div style={{ color: 'var(--text-muted)', fontSize: '13px' }} className="mono">
                    Loading expansion parameters...
                  </div>
                )}
              </div>
            </GlassCard>

            {/* Sector Job Drift Heatmap */}
            <GlassCard title="Sector Recruitment Drift" glowType="blue">
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', marginTop: '10px' }}>
                {sectorStats.length > 0 ? sectorStats.map(sector => (
                  <div key={sector.name} style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <div style={{ width: '100px', fontSize: '12px', color: 'var(--text-muted)', textTransform: 'capitalize' }}>
                      {sector.name}
                    </div>
                    <div style={{ flex: 1, height: '16px', background: 'rgba(255,255,255,0.05)', borderRadius: '4px', position: 'relative', overflow: 'hidden' }}>
                      <div style={{
                        width: `${Math.min(sector.count * 15, 100)}%`,
                        height: '100%',
                        background: 'linear-gradient(90deg, var(--blue), var(--cyan))',
                        borderRadius: '4px'
                      }} />
                      <span className="mono" style={{ position: 'absolute', right: '8px', top: '50%', transform: 'translateY(-50%)', fontSize: '10px', color: 'var(--text-primary)', fontWeight: 600 }}>
                        {sector.count} Jobs
                      </span>
                    </div>
                  </div>
                )) : (
                  <div style={{ color: 'var(--text-muted)', fontSize: '13px' }} className="mono">
                    Aggregating recruitment telemetry...
                  </div>
                )}
              </div>
            </GlassCard>
          </div>

          {/* Live Narrative Feed */}
          {narratives.length > 0 && (
            <GlassCard title="💡 Live Narrative Feed" glowType="cyan" style={{ marginBottom: 30 }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', marginTop: '10px' }}>
                {narratives.map((nar, idx) => (
                  <div key={idx} style={{ 
                    padding: '12px', 
                    background: 'rgba(255,255,255,0.02)', 
                    borderLeft: '3px solid var(--cyan)', 
                    borderRadius: '0 8px 8px 0' 
                  }}>
                    <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '4px' }}>
                      {nar.summary}
                    </div>
                    <div style={{ fontSize: '11px', color: 'var(--text-muted)' }} className="mono">
                      Timeframe: {nar.timeframe} | Recommendation: {nar.recommendation}
                    </div>
                  </div>
                ))}
              </div>
            </GlassCard>
          )}
        </div>

        <div style={{ flex: '1 1 300px', minWidth: '300px', display: 'flex', flexDirection: 'column' }}>
          <NewsPanel domain="all" title="Global Telemetry Intel" />
        </div>
      </div>
    </div>
  )
}

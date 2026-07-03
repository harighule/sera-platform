import { useEffect, useState, useRef } from 'react'
import { fetchStats } from '../api/client'
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

    // Poll stats
    const interval = setInterval(() => {
      fetchStats().then(data => {
        if (data) {
          setStats(data)
          
          // Toast alert if active alerts count increases
          if (data.active_alerts > prevAlertsRef.current) {
            addToast(`AXIOM-Φ: ${data.active_alerts - prevAlertsRef.current} new pre-transition behavior alerts detected!`, 'critical')
          }
          prevAlertsRef.current = data.active_alerts ?? 0

          // Append to chart history
          setHistory(prev => [
            ...prev,
            {
              time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
              eps: parseFloat(data.events_per_second) || 0,
              entropy: parseFloat(data.entropy_average) || 0
            }
          ].slice(-15)) // Keep last 15 ticks
        }
      })
    }, 5000)

    // Listen to WebSocket Event Stream
    wsRef.current = createStream((data) => {
      if (data.event) {
        const ev = data.event
        setFeed(prev => [ev, ...prev].slice(0, 30))
        
        // Show success toast on learning or evolve logs
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
    <div style={{ animation: 'fadeUp 0.4s ease', display: 'flex', gap: '24px', flexWrap: 'wrap', alignItems: 'stretch' }}>
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

        <div className="grid-2" style={{ marginBottom: 30 }}>
          {/* Visual Chart Card */}
          <GlassCard title="Telemetric Flow & Threat Potential" glowType="cyan">
            <div style={{ height: 260, marginTop: 10 }}>
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
            <div style={{ display: 'flex', gap: 16, fontSize: 11, color: 'var(--text-muted)', marginTop: 12, justifyContent: 'center' }} className="mono">
              <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span style={{ width: 8, height: 8, background: 'var(--cyan)', borderRadius: '50%' }} /> Ingestion EPS
              </span>
              <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span style={{ width: 8, height: 8, background: 'var(--blue)', borderRadius: '50%' }} /> Average Entropy (AXIOM-Φ)
              </span>
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
                  <span style={{ color: 'var(--text-dimmed)', marginLeft: 'auto', fontSize: 11 }} className="mono">
                    {new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                  </span>
                </div>
              ))}
            </div>
          </GlassCard>
        </div>
      </div>

      <div style={{ flex: '1 1 300px', minWidth: '300px', display: 'flex', flexDirection: 'column' }}>
        <NewsPanel domain="all" title="Global Telemetry Intel" />
      </div>
    </div>
  )
}

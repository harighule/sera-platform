import { useEffect, useState } from 'react'
import { fetchEntropy, fetchAlerts } from '../api/client'
import GlassCard from '../components/GlassCard'
import AnimatedCounter from '../components/AnimatedCounter'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import NewsPanel from '../components/NewsPanel'

export default function AxiomMonitor() {
  const [entropy, setEntropy] = useState([])
  const [alerts, setAlerts] = useState([])
  const [selectedEntity, setSelectedEntity] = useState(null)

  useEffect(() => {
    fetchEntropy().then(data => {
      setEntropy(data)
      setSelectedEntity(prev => prev || (data.length > 0 ? data[0] : null))
    })
    fetchAlerts().then(setAlerts)

    const i = setInterval(() => {
      fetchEntropy().then(data => {
        setEntropy(data)
        setSelectedEntity(prev => {
          if (!prev) return data.length > 0 ? data[0] : null
          const updated = data.find(item => item.entity_name === prev.entity_name)
          return updated || prev
        })
      })
      fetchAlerts().then(setAlerts)
    }, 4000)
    return () => clearInterval(i)
  }, [])

  useEffect(() => {
    if (selectedEntity) {
      console.log(`Targeting node entropy signature for: ${selectedEntity.entity_name}`)
    }
  }, [selectedEntity])

  // Get color for entropy score
  const getEntropyColor = (val) => {
    if (val > 2.2) return 'var(--red)'
    if (val > 1.4) return 'var(--amber)'
    return 'var(--cyan)'
  }

  // Get glow style based on status
  const getStatusGlow = (status) => {
    if (status === 'critical') return 'red'
    if (status === 'pre-transition') return 'amber'
    return 'cyan'
  }

  // Prep data for Recharts (Top 8 highest entropy)
  const chartData = [...entropy]
    .sort((a, b) => b.entropy - a.entropy)
    .slice(0, 8)
    .map(item => ({
      name: item.entity_name.split(' ')[0], // short name
      fullName: item.entity_name,
      entropy: parseFloat(item.entropy) || 0,
      status: item.status
    }))

  return (
    <div style={{ animation: 'fadeUp 0.4s ease', display: 'flex', gap: '24px', flexWrap: 'wrap', alignItems: 'stretch' }} className="axiom-monitor">
      <div style={{ flex: '3 1 350px', minWidth: '350px' }}>
      <div className="grid-2" style={{ marginBottom: 24 }}>
        <GlassCard glowType={alerts.length > 0 ? 'red' : ''}>
          <div className="stat-label">Active Pre-Transition Alerts</div>
          <div className="stat-value mono" style={{ color: alerts.length > 0 ? 'var(--red)' : 'var(--text-primary)' }}>
            <AnimatedCounter value={alerts.length} />
          </div>
          <div className="stat-sub">Spike detections (z-score &gt; 2.0)</div>
        </GlassCard>
        <GlassCard glowType="cyan">
          <div className="stat-label">Entities Monitored</div>
          <div className="stat-value mono">
            <AnimatedCounter value={entropy.length} />
          </div>
          <div className="stat-sub">AXIOM-Φ state trackers active</div>
        </GlassCard>
      </div>

      {/* Alerts Area */}
      {alerts.length > 0 && (
        <div style={{ marginBottom: 24 }}>
          <GlassCard title="⚠ CRITICAL PRE-TRANSITION SIRENS" glowType="red">
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {alerts.map((a, idx) => (
                <div 
                  key={idx} 
                  style={{ 
                    padding: '14px 18px', 
                    background: 'rgba(255, 0, 60, 0.04)', 
                    border: '1px solid rgba(255, 0, 60, 0.15)', 
                    borderRadius: 8,
                    display: 'flex',
                    flexWrap: 'wrap',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    gap: 12
                  }}
                  className="mono animate-pulse-glow"
                >
                  <div>
                    <span style={{ fontWeight: 800, color: 'var(--red)', fontSize: 14 }}>{a.entity_name}</span>
                    <span className="badge badge-red" style={{ marginLeft: 12 }}>{a.severity}</span>
                    <div style={{ color: 'var(--text-secondary)', fontSize: 12, marginTop: 4 }}>{a.description}</div>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>SHANNON ENTROPY</div>
                    <div style={{ fontSize: 18, fontWeight: 'bold', color: 'var(--red)' }}>
                      {a.entropy_value?.toFixed(4)}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </GlassCard>
        </div>
      )}

      <div className="grid-2">
        {/* Left Column: Top entropy chart & Detailed inspector */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
          {/* Recharts Bar Comparison */}
          <GlassCard title="Entropy Variance - High-Risk Entities" glowType="cyan">
            <div style={{ height: 200, marginTop: 10 }}>
              {chartData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                    <XAxis dataKey="name" stroke="var(--text-muted)" fontSize={9} tickLine={false} />
                    <YAxis stroke="var(--text-muted)" fontSize={9} tickLine={false} />
                    <Tooltip
                      content={({ active, payload }) => {
                        if (active && payload && payload.length) {
                          const data = payload[0].payload
                          return (
                            <div style={{ background: 'rgba(10, 15, 30, 0.95)', border: '1px solid var(--border)', padding: '10px 12px', borderRadius: 8, fontSize: 12 }}>
                              <div style={{ fontWeight: 'bold', color: 'var(--text-primary)', marginBottom: 4 }}>{data.fullName}</div>
                              <div>Entropy: <span className="mono" style={{ color: getEntropyColor(data.entropy), fontWeight: 'bold' }}>{data.entropy.toFixed(4)}</span></div>
                              <div>Status: <span style={{ textTransform: 'capitalize' }}>{data.status}</span></div>
                            </div>
                          )
                        }
                        return null
                      }}
                    />
                    <Bar dataKey="entropy" radius={[4, 4, 0, 0]}>
                      {chartData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={getEntropyColor(entry.entropy)} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div style={{ color: 'var(--text-muted)', fontSize: 12, textAlign: 'center', paddingTop: 80 }}>Gathering entropy registers...</div>
              )}
            </div>
          </GlassCard>

          {/* Inspector Panel */}
          {selectedEntity && (
            <GlassCard title="Entity Entropy Inspector" glowType={getStatusGlow(selectedEntity.status)}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <h4 style={{ fontSize: 18, fontWeight: 700, color: 'var(--text-primary)' }}>{selectedEntity.entity_name}</h4>
                  <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>
                    Registry Reference Status: <span style={{ color: getEntropyColor(selectedEntity.entropy), fontWeight: 'bold', textTransform: 'capitalize' }}>{selectedEntity.status}</span>
                  </div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontSize: 9, color: 'var(--text-muted)' }}>CURRENT SHANNON INDEX</div>
                  <div className="mono" style={{ fontSize: 26, fontWeight: 800, color: getEntropyColor(selectedEntity.entropy) }}>
                    {selectedEntity.entropy?.toFixed(5)}
                  </div>
                </div>
              </div>
              
              {/* Stability gauge bar indicator */}
              <div style={{ marginTop: 16 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--text-muted)', marginBottom: 4 }}>
                  <span>STABILITY THRESHOLD</span>
                  <span>{selectedEntity.entropy > 2.2 ? 'STATE DECAY DETECTED' : 'STEADY CONFIGURATION'}</span>
                </div>
                <div style={{ width: '100%', height: 6, background: 'rgba(255,255,255,0.05)', borderRadius: 3, overflow: 'hidden' }}>
                  <div 
                    style={{ 
                      height: '100%', 
                      width: `${Math.min((selectedEntity.entropy / 4.0) * 100, 100)}%`, 
                      background: getEntropyColor(selectedEntity.entropy),
                      boxShadow: `0 0 10px ${getEntropyColor(selectedEntity.entropy)}`,
                      transition: 'width 0.4s ease'
                    }} 
                  />
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9, color: 'var(--text-dimmed)', marginTop: 4 }} className="mono">
                  <span>0.0 (Null Signal)</span>
                  <span>2.0 (Alert Boundary)</span>
                  <span>4.0 (Turbulent Chaos)</span>
                </div>
              </div>
            </GlassCard>
          )}
        </div>

        {/* Right Column: 10x5 Dense Entropy Heatmap Grid */}
        <GlassCard title="Global Entropy Heatmap (50 resolved registry nodes)" glowType="blue">
          <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 16, lineHeight: '1.4' }}>
            Hover cells to query node entropy signatures. Click to lock node target inside inspector.
          </div>
          
          {/* Heatmap Grid */}
          <div 
            style={{ 
              display: 'grid', 
              gridTemplateColumns: 'repeat(10, 1fr)', 
              gap: 8, 
              aspectRatio: '2/1', 
              marginBottom: 16 
            }}
          >
            {entropy.map((e, idx) => {
              const color = getEntropyColor(e.entropy)
              const isSelected = selectedEntity && selectedEntity.entity_name === e.entity_name
              
              return (
                <div
                  key={idx}
                  onClick={() => setSelectedEntity(e)}
                  style={{
                    background: color,
                    opacity: isSelected ? 1 : 0.45,
                    borderRadius: 4,
                    cursor: 'pointer',
                    transition: 'all 0.2s ease',
                    boxShadow: isSelected ? `0 0 12px ${color}` : 'none',
                    border: isSelected ? '1.5px solid var(--text-primary)' : '1px solid transparent'
                  }}
                  title={`${e.entity_name}: ${e.entropy?.toFixed(4)} (${e.status})`}
                />
              )}
            )}
            
            {/* Pad grid if registry incomplete */}
            {Array.from({ length: Math.max(0, 50 - entropy.length) }).map((_, idx) => (
              <div 
                key={`empty-${idx}`} 
                style={{ background: 'rgba(255,255,255,0.02)', borderRadius: 4, border: '1px dashed rgba(255,255,255,0.05)' }} 
              />
            ))}
          </div>
          
          {/* Heatmap Legend */}
          <div style={{ display: 'flex', gap: 16, fontSize: 10, color: 'var(--text-muted)', justifyContent: 'center' }} className="mono">
            <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ width: 10, height: 10, background: 'var(--cyan)', borderRadius: 2 }} /> Stable Entropy (&lt; 1.4)
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ width: 10, height: 10, background: 'var(--amber)', borderRadius: 2 }} /> Medium Entropy (1.4 - 2.2)
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ width: 10, height: 10, background: 'var(--red)', borderRadius: 2 }} /> Turbulence Spikes (&gt; 2.2)
            </span>
          </div>
        </GlassCard>
      </div>
      </div>
      <div style={{ flex: '1 1 300px', minWidth: '300px', display: 'flex', flexDirection: 'column' }}>
        <NewsPanel 
          domain={selectedEntity?.domain || 'healthcare'} 
          title={`${selectedEntity?.entity_name?.split(' ')[0] || 'System'} Telemetry News`} 
        />
      </div>
    </div>
  )
}
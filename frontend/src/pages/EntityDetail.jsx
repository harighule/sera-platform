import { useEffect, useState, useRef } from 'react'
import { useParams, Link } from 'react-router-dom'
import { fetchEntityFullProfile, sendChat } from '../api/client'
import GlassCard from '../components/GlassCard'
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip } from 'recharts'

const NetworkGraph = ({ data }) => {
  if (!data || !data.nodes) return <div className="mono text-muted">No graph data</div>;

  const width = 500;
  const height = 300;
  const cx = width / 2;
  const cy = height / 2;
  const radius = 110;

  // Lay out target nodes in a circle
  const nodes = data.nodes.map((node, index) => {
    if (index === 0) {
      return { ...node, x: cx, y: cy, isCenter: true };
    }
    const angle = ((index - 1) / (data.nodes.length - 1)) * 2 * Math.PI;
    return {
      ...node,
      x: cx + radius * Math.cos(angle),
      y: cy + radius * Math.sin(angle),
      isCenter: false
    };
  });

  return (
    <div style={{ position: 'relative', width: '100%', height: '300px' }}>
      <svg width="100%" height="100%" viewBox={`0 0 ${width} ${height}`}>
        <defs>
          <radialGradient id="glow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="var(--cyan)" stopOpacity="0.4" />
            <stop offset="100%" stopColor="var(--cyan)" stopOpacity="0" />
          </radialGradient>
        </defs>

        {/* Outer connection ring */}
        <circle cx={cx} cy={cy} r={radius} fill="none" stroke="rgba(255,255,255,0.03)" strokeWidth={1} />

        {/* Edge Lines */}
        {data.edges.map((edge, idx) => {
          const sourceNode = nodes.find(n => n.id === edge.source);
          const targetNode = nodes.find(n => n.id === edge.target);
          if (!sourceNode || !targetNode) return null;
          return (
            <line
              key={idx}
              x1={sourceNode.x}
              y1={sourceNode.y}
              x2={targetNode.x}
              y2={targetNode.y}
              stroke="var(--cyan)"
              strokeWidth={Math.min(edge.weight * 0.8 + 1, 4)}
              strokeOpacity={0.4 + (edge.weight * 0.1)}
              strokeDasharray="5 3"
              style={{ animation: 'dash 10s linear infinite' }}
            />
          );
        })}

        {/* Center Node Glow */}
        <circle cx={cx} cy={cy} r={35} fill="url(#glow)" />

        {/* Nodes */}
        {nodes.map(node => (
          <g key={node.id} style={{ cursor: 'pointer' }}>
            <circle
              cx={node.x}
              cy={node.y}
              r={node.isCenter ? 18 : 10}
              fill={node.isCenter ? 'var(--cyan)' : 'var(--bg)'}
              stroke={node.isCenter ? 'var(--bg-card)' : 'var(--cyan)'}
              strokeWidth={2}
            />
            <text
              x={node.x}
              y={node.y + (node.isCenter ? 28 : 20)}
              textAnchor="middle"
              fill={node.isCenter ? 'var(--text-primary)' : 'var(--text-secondary)'}
              fontSize={node.isCenter ? 12 : 9}
              fontWeight={node.isCenter ? 'bold' : 'normal'}
              className="mono"
            >
              {node.id}
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
};

export default function EntityDetail() {
  const { ticker } = useParams()
  const [profile, setProfile] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(false)
  const [chatInput, setChatInput] = useState('')
  const [chatLog, setChatLog] = useState([])
  const [chatLoading, setChatLoading] = useState(false)
  const chatBottomRef = useRef(null)

  useEffect(() => {
    async function load() {
      setLoading(true)
      setError(false)
      const res = await fetchEntityFullProfile(ticker)
      if (res && res.ticker) {
        setProfile(res)
        setChatLog([
          {
            role: 'ai',
            text: `SYSTEM ACTIVE. Intelligence profile loaded for ${res.company_name} (${res.ticker}). What insights would you like to request?`
          }
        ])
      } else {
        setError(true)
      }
      setLoading(false)
    }
    load()
  }, [ticker])

  useEffect(() => {
    if (chatBottomRef.current) {
      chatBottomRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [chatLog])

  if (loading) {
    return (
      <div className="flex-center" style={{ height: '70vh', flexDirection: 'column', gap: '16px' }}>
        <div className="spinner"></div>
        <p className="mono text-muted">Retrieving multi-source causal signals...</p>
      </div>
    )
  }

  if (error || !profile) {
    return (
      <div className="flex-center" style={{ height: '70vh', flexDirection: 'column', gap: '24px' }}>
        <div style={{ color: 'var(--red)', fontSize: '48px' }}>⚠️</div>
        <h2 className="mono">Entity Profile Not Found</h2>
        <p className="text-muted" style={{ maxWidth: '400px', textAlign: 'center' }}>
          We could not resolve telemetry signatures or database records matching the ticker "{ticker}".
        </p>
        <Link to="/entities" className="btn cyan font-bold mono">Return to Registry</Link>
      </div>
    )
  }

  const handleSendChat = async (e) => {
    e.preventDefault()
    if (!chatInput.trim() || chatLoading) return

    const userText = chatInput
    setChatLog(prev => [...prev, { role: 'user', text: userText }])
    setChatInput('')
    setChatLoading(true)

    try {
      // Prepend AI context prompt in the final payload to guide response
      const payload = `${profile.ai_context}\n\nUser Question: ${userText}`
      const res = await sendChat(payload)
      setChatLog(prev => [...prev, { role: 'ai', text: res.response }])
    } catch {
      setChatLog(prev => [...prev, { role: 'ai', text: 'CONNECTION FAILED: Chat endpoint timed out.' }])
    }
    setChatLoading(false)
  }

  // Construct mock timeline charts for AXIOM monitor
  const entropyHistory = [
    { time: 'T-24h', entropy: Math.max(0.1, profile.axiom_entropy.current_entropy - 0.12) },
    { time: 'T-18h', entropy: Math.max(0.1, profile.axiom_entropy.current_entropy - 0.08) },
    { time: 'T-12h', entropy: Math.max(0.1, profile.axiom_entropy.current_entropy + 0.03) },
    { time: 'T-6h', entropy: Math.max(0.1, profile.axiom_entropy.current_entropy - 0.05) },
    { time: 'Current', entropy: profile.axiom_entropy.current_entropy }
  ];

  return (
    <div style={{ animation: 'fadeUp 0.3s ease', display: 'flex', flexDirection: 'column', gap: '24px' }}>
      
      {/* Back button and title */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Link to="/entities" className="mono btn" style={{ padding: '6px 12px', fontSize: '12px' }}>
          ← Back to Registry
        </Link>
        <div className="tag mono cyan">CALIBRATED FEED: ACTIVE</div>
      </div>

      <div style={{ display: 'flex', alignItems: 'baseline', gap: '12px' }}>
        <h1 style={{ margin: 0, fontSize: '2.5rem' }}>{profile.company_name}</h1>
        <span className="mono text-muted" style={{ fontSize: '1.25rem' }}>({profile.ticker})</span>
      </div>

      {/* Grid Layout: 2 Column Desktop, 1 Column Mobile */}
      <div className="dashboard-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(450px, 1fr))', gap: '24px' }}>
        
        {/* WIDGET 1: Entity Graph */}
        <GlassCard title="Entity Relationship Graph (1-Hop)" glowType="cyan">
          <p className="text-muted font-sm" style={{ marginBottom: '12px' }}>
            Telemetry co-occurrences mapping dynamic sector links and partners.
          </p>
          <NetworkGraph data={profile.network_graph} />
        </GlassCard>

        {/* WIDGET 2: Claim Credibility */}
        <GlassCard title="Claim Credibility Index (ALETHEIA)" glowType="emerald">
          <div style={{ display: 'flex', alignItems: 'center', gap: '32px', margin: '20px 0' }}>
            <div style={{ position: 'relative', width: '100px', height: '100px', flexShrink: 0 }}>
              {/* Circular Gauge */}
              <svg width="100" height="100" viewBox="0 0 36 36">
                <path
                  d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                  fill="none"
                  stroke="rgba(255,255,255,0.03)"
                  strokeWidth="3"
                />
                <path
                  d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                  fill="none"
                  stroke="var(--emerald)"
                  strokeWidth="3"
                  strokeDasharray={`${profile.credibility.score}, 100`}
                />
              </svg>
              <div style={{
                position: 'absolute',
                top: 0, left: 0, right: 0, bottom: 0,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                flexDirection: 'column'
              }}>
                <span className="mono font-bold" style={{ fontSize: '20px' }}>{profile.credibility.score}%</span>
                <span className="mono text-muted" style={{ fontSize: '8px' }}>CRED</span>
              </div>
            </div>
            
            <div>
              <h3 className="mono font-md" style={{ margin: '0 0 8px 0', color: 'var(--emerald)' }}>Credibility Profile</h3>
              <ul style={{ paddingLeft: '16px', margin: 0 }}>
                {profile.credibility.factors.map((f, i) => (
                  <li key={i} className="font-sm text-secondary" style={{ marginBottom: '4px' }}>{f}</li>
                ))}
              </ul>
            </div>
          </div>
        </GlassCard>

        {/* WIDGET 3: Citation Tracking */}
        <GlassCard title="Citation Timeline Monitor" glowType="cyan" style={{ maxHeight: '350px', overflowY: 'auto' }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            {profile.citations.map((c, idx) => (
              <div key={c.id || idx} style={{ borderLeft: '2px solid var(--border)', paddingLeft: '16px', position: 'relative' }}>
                <div style={{
                  position: 'absolute',
                  left: '-6px',
                  top: '4px',
                  width: '10px',
                  height: '10px',
                  borderRadius: '50%',
                  background: 'var(--cyan)'
                }}></div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: '4px' }}>
                  <span className="mono font-sm font-bold text-primary">{c.source}</span>
                  <span className="mono text-muted" style={{ fontSize: '10px' }}>{c.date ? c.date.split('T')[0] : 'N/A'}</span>
                </div>
                <h4 className="font-sm" style={{ margin: '0 0 4px 0' }}>{c.title}</h4>
                <p className="text-secondary font-xs" style={{ margin: 0 }}>{c.summary}</p>
                <div style={{ marginTop: '4px' }}>
                  <span className={`tag mono ${c.tone >= 0 ? 'green-tag' : 'red-tag'}`} style={{ fontSize: '9px', padding: '2px 6px' }}>
                    Tone: {c.tone}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </GlassCard>

        {/* WIDGET 4: AXIOM-Φ Monitor */}
        <GlassCard title="AXIOM-Φ Entropy Sparkline" glowType="blue">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <div>
              <span className="mono text-muted font-sm">CURRENT ENTROPY</span>
              <div className="mono font-lg font-bold" style={{ color: 'var(--blue)' }}>{profile.axiom_entropy.current_entropy}</div>
            </div>
            <div style={{ textAlign: 'right' }}>
              <span className="mono text-muted font-sm">STATUS REGISTER</span>
              <div className={`tag mono ${profile.axiom_entropy.status === 'STABLE' ? 'green-tag' : 'red-tag'}`}>
                {profile.axiom_entropy.status}
              </div>
            </div>
          </div>
          <div style={{ width: '100%', height: '180px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={entropyHistory}>
                <defs>
                  <linearGradient id="colorEntropy" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="var(--blue)" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="var(--blue)" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <XAxis dataKey="time" stroke="rgba(255,255,255,0.2)" fontSize={9} />
                <YAxis stroke="rgba(255,255,255,0.2)" fontSize={9} />
                <Tooltip contentStyle={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }} />
                <Area type="monotone" dataKey="entropy" stroke="var(--blue)" fillOpacity={1} fill="url(#colorEntropy)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </GlassCard>

        {/* WIDGET 5: ZOLA Predictions */}
        <GlassCard title="ZOLA Predictive Outcomes" glowType="cyan">
          <p className="text-muted font-sm" style={{ marginBottom: '20px' }}>
            Deep neural inference estimates of operational risk and corporate movement.
          </p>
          <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap' }}>
            <div className="panel" style={{ flex: '1 1 120px', padding: '16px', background: 'rgba(255,255,255,0.01)', border: '1px solid var(--border)' }}>
              <span className="mono text-muted font-xs">EXPANSION PROBABILITY</span>
              <div className="mono font-lg font-bold text-primary" style={{ marginTop: '8px' }}>
                {(profile.predictions.expansion_score * 100).toFixed(1)}%
              </div>
            </div>
            <div className="panel" style={{ flex: '1 1 120px', padding: '16px', background: 'rgba(255,255,255,0.01)', border: '1px solid var(--border)' }}>
              <span className="mono text-muted font-xs">PURCHASE INTENT SCORE</span>
              <div className="mono font-lg font-bold text-primary" style={{ marginTop: '8px' }}>
                {(profile.predictions.purchase_intent * 100).toFixed(1)}%
              </div>
            </div>
            <div className="panel" style={{ flex: '1 1 120px', padding: '16px', background: 'rgba(255,255,255,0.01)', border: '1px solid var(--border)' }}>
              <span className="mono text-muted font-xs">SUPPLY CHAIN RISK</span>
              <div className="mono font-lg font-bold text-primary" style={{ marginTop: '8px' }}>
                {(profile.predictions.supply_chain_risk * 100).toFixed(1)}%
              </div>
            </div>
          </div>
        </GlassCard>

        {/* WIDGET 6: AI Command */}
        <GlassCard title="Entity Analysis Chat Console" glowType="cyan" style={{ display: 'flex', flexDirection: 'column', height: '350px' }}>
          <div style={{ flex: 1, overflowY: 'auto', padding: '8px', display: 'flex', flexDirection: 'column', gap: '12px', borderBottom: '1px solid var(--border)' }}>
            {chatLog.map((c, idx) => (
              <div key={idx} style={{ display: 'flex', flexDirection: 'column', alignSelf: c.role === 'user' ? 'flex-end' : 'flex-start', maxWidth: '85%' }}>
                <span className="mono text-muted font-xs" style={{ marginBottom: '2px', textAlign: c.role === 'user' ? 'right' : 'left' }}>
                  {c.role === 'user' ? 'TACTICAL_OPERATOR' : 'SERA_AGI_CORE'}
                </span>
                <div style={{
                  padding: '10px 14px',
                  borderRadius: '12px',
                  background: c.role === 'user' ? 'rgba(6, 182, 212, 0.15)' : 'rgba(255, 255, 255, 0.02)',
                  border: c.role === 'user' ? '1px solid var(--cyan)' : '1px solid var(--border)'
                }}>
                  <p className="font-sm" style={{ margin: 0, lineHeight: 1.4 }}>{c.text}</p>
                </div>
              </div>
            ))}
            {chatLoading && <div className="mono text-muted font-xs">Processing prompt parameters...</div>}
            <div ref={chatBottomRef} />
          </div>

          <form onSubmit={handleSendChat} style={{ display: 'flex', gap: '8px', padding: '12px 0 0 0' }}>
            <input
              type="text"
              className="mono"
              placeholder="Query corporate metrics..."
              value={chatInput}
              onChange={e => setChatInput(e.target.value)}
              style={{ flex: 1, background: 'rgba(0,0,0,0.3)', border: '1px solid var(--border)', color: 'var(--text-primary)', padding: '10px', borderRadius: '4px' }}
            />
            <button type="submit" className="btn cyan font-bold mono">SEND</button>
          </form>
        </GlassCard>

        {/* WIDGET 7: Dark Intel */}
        <GlassCard title="Dark Intel Vulnerability briefing" glowType="red" style={{ borderColor: 'rgba(239, 68, 68, 0.2)' }}>
          <p className="text-muted font-sm" style={{ marginBottom: '16px' }}>
            Threat briefings co-occurring with cyberspace leaks or data breaches.
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {profile.dark_intel.map((d, i) => (
              <div key={d.id || i} style={{
                background: 'rgba(239, 68, 68, 0.02)',
                border: '1px solid rgba(239, 68, 68, 0.1)',
                borderRadius: '6px',
                padding: '12px',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center'
              }}>
                <div>
                  <h4 className="font-sm" style={{ margin: '0 0 4px 0', color: d.severity === 'critical' ? 'var(--red)' : 'var(--text-primary)' }}>
                    {d.title}
                  </h4>
                  <span className="mono text-muted font-xs">DETECTED: {d.detected_at ? d.detected_at.split('T')[0] : 'N/A'}</span>
                </div>
                <span className={`tag mono ${d.severity === 'critical' || d.severity === 'high' ? 'red-tag' : 'yellow-tag'}`}>
                  {d.severity.toUpperCase()}
                </span>
              </div>
            ))}
          </div>
        </GlassCard>

      </div>
    </div>
  )
}

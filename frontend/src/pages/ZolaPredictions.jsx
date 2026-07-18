import { useEffect, useState, useRef } from 'react'
import {
  fetchZolaDashboard,
  fetchZolaStatus,
  triggerCyberspaceLearning,
  proposeSelfEvolution,
  validateSelfEvolution,
  approveSelfEvolution,
  runKronosOptimize,
  fetchKronosStatus,
  fetchEntityArchitecture,
  fetchAxiomAnalysis,
  triggerKronosScaling,
  getScalingStatus,
  runAxiomCompression,
  getGodelAutoStatus
} from '../api/client'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend
} from 'recharts'
import GlassCard from '../components/GlassCard'
import AnimatedCounter from '../components/AnimatedCounter'
import { useToast } from '../components/ToastNotification'
import NewsPanel from '../components/NewsPanel'

export default function ZolaPredictions() {
  const [predictions, setPredictions] = useState([])
  const [activeTab, setActiveTab] = useState('predictions') // 'predictions' or 'kronos'
  const [status, setStatus] = useState(null)
  const [kronosStatus, setKronosStatus] = useState(null)
  const [actionLog, setActionLog] = useState([])
  const [loading, setLoading] = useState(false)
  const [optimizationHistory, setOptimizationHistory] = useState([])
  const [isOptimizing, setIsOptimizing] = useState(false)
  const [architectureReport, setArchitectureReport] = useState(null)
  const [axiomReport, setAxiomReport] = useState(null)
  // Gödel scaling & AXIOM compression state
  const [scalingResult, setScalingResult] = useState(null)
  const [scalingStatus, setScalingStatus] = useState(null)
  const [compressionResult, setCompressionResult] = useState(null)
  const [godelStatus, setGodelStatus] = useState(null)
  const [isScaling, setIsScaling] = useState(false)
  const [isCompressing, setIsCompressing] = useState(false)
  const terminalEndRef = useRef(null)
  const { addToast } = useToast()

  // Load KRONOS status
  useEffect(() => {
    fetchKronosStatus().then(setKronosStatus).catch(() => {})
  }, [])

  // Load architecture report and AXIOM analysis once on mount
  useEffect(() => {
    fetchEntityArchitecture().then(setArchitectureReport).catch(() => {})
    fetchAxiomAnalysis().then(setAxiomReport).catch(() => {})
    getScalingStatus().then(setScalingStatus).catch(() => {})
    getGodelAutoStatus().then(setGodelStatus).catch(() => {})
  }, [])

  // Poll scaling + Gödel status every 15 seconds
  useEffect(() => {
    const id = setInterval(() => {
      getScalingStatus().then(setScalingStatus).catch(() => {})
      getGodelAutoStatus().then(setGodelStatus).catch(() => {})
    }, 15000)
    return () => clearInterval(id)
  }, [])

  // Load predictions
  const loadDashboard = () => {
    fetchZolaDashboard().then(data => {
      if (data && data.predictions) {
        setPredictions(data.predictions)
      }
    }).catch(() => {})
  }

  useEffect(() => {
    loadDashboard()
    const i = setInterval(loadDashboard, 8000)
    return () => clearInterval(i)
  }, [])

  // Load ZOLA status details
  const refreshStatus = () => {
    fetchZolaStatus().then(setStatus).catch(() => {})
  }

  useEffect(() => {
    refreshStatus()
    const i = setInterval(refreshStatus, 8000)
    return () => clearInterval(i)
  }, [])

  // Auto-scroll main terminal
  useEffect(() => {
    if (terminalEndRef.current) {
      terminalEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [actionLog])

  // Auto-scroll optimization log
  const optScrollRef = useRef(null)
  useEffect(() => {
    if (optScrollRef.current) {
      optScrollRef.current.scrollTop = optScrollRef.current.scrollHeight
    }
  }, [optimizationHistory])

  // Scroll KRONOS tab into view on switch
  const kronosRef = useRef(null)
  useEffect(() => {
    if (activeTab === 'kronos' && kronosRef.current) {
      kronosRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }, [activeTab])

  const logAction = (msg) => {
    setActionLog(prev => [...prev, `[${new Date().toLocaleTimeString()}] ${msg}`].slice(-25))
  }

  const handleLearn = async () => {
    setLoading(true)
    logAction("INITIATING: Cyberspace crawler daemon...")
    addToast("KRONOS: Cyberspace crawling starting...", "info")
    try {
      const res = await triggerCyberspaceLearning()
      logAction(`SUCCESS: Cyberspace crawl completed. Ingested new entity facts.`)
      logAction(`KRONOS: PyTorch weights scaled to ${res.parameter_scale.toLocaleString()} parameters.`);
      addToast("KRONOS: Cyberspace learning completed. Parameters updated.", "success")
      refreshStatus()
    } catch (err) {
      logAction("ERROR: Cyberspace learning requires LIVE Entity mode.")
      addToast("Cyberspace learning failed. Server returned error.", "critical")
    }
    setLoading(false)
  }

  const handlePropose = async () => {
    setLoading(true)
    logAction("PROPOSING: Proposing code self-rewriting optimization patch...")
    addToast("KRONOS: Proposing evolution patch...", "info")
    try {
      const res = await proposeSelfEvolution()
      logAction(`PROPOSED: Patch #${res.patch_id} proposed successfully for target: ${res.target_file}`)
      addToast(`Patch #${res.patch_id} proposed for optimization!`, "warning")
      refreshStatus()
    } catch (err) {
      logAction("ERROR: Self-evolution proposal requires LIVE Entity mode.")
      addToast("Patch proposal failed. Check active system mode.", "critical")
    }
    setLoading(false)
  }

  const handleValidate = async (patchId) => {
    setLoading(true)
    logAction(`COMPILING: Validating Patch #${patchId} in secure sandbox compiler...`)
    addToast(`Compiling Patch #${patchId} in sandbox...`, "info")
    try {
      const res = await validateSelfEvolution(patchId)
      if (res.status === 'success') {
        logAction(`VALIDATION PASSED: Sandbox compile successful! Constraints verified. Regressions: 0.`)
        addToast(`Patch #${patchId} syntax & safety checks passed!`, "success")
      } else {
        logAction(`VALIDATION FAILED: Compile error or safety constraint violated!`)
        addToast(`Patch #${patchId} validation failed!`, "critical")
      }
      refreshStatus()
    } catch (err) {
      logAction("ERROR: Sandbox compile execution interrupted.")
      addToast("Validation execution failed.", "critical")
    }
    setLoading(false)
  }

  const handleApprove = async (patchId) => {
    setLoading(true)
    logAction(`APPLYING: Dynamically hot-patching runtime in-memory logic...`)
    addToast(`Applying Patch #${patchId} dynamically...`, "info")
    try {
      await approveSelfEvolution(patchId)
      logAction(`SUCCESS: Patch #${patchId} applied successfully. Parameters hot-swapped without system restart.`)
      addToast(`Patch #${patchId} applied and optimized!`, "success")
      refreshStatus()
    } catch (err) {
      logAction("ERROR: Failed to apply in-memory code patch.")
      addToast("Hot patching failed.", "critical")
    }
    setLoading(false)
  }

  const handleRunOptimization = async () => {
    setIsOptimizing(true)
    logAction('KRONOS: Firing differentiable optimization step...')
    try {
      const res = await runKronosOptimize()
      setOptimizationHistory(prev => [...prev, res].slice(-20))
      logAction(`SUCCESS: Backprop step #${res.backprop_steps} — loss=${res.latest_loss}, grad_norm=${res.latest_grad_norm}, latency=${res.latency_ms}ms`)
    } catch (err) {
      logAction(`ERROR: ${err.message ?? 'Optimization step failed. Requires ENTITY_MODE=live.'}`)
    }
    setIsOptimizing(false)
  }

  const handleRunScaling = async () => {
    setIsScaling(true)
    try {
      const r = await triggerKronosScaling()
      setScalingResult(r)
      getScalingStatus().then(setScalingStatus).catch(() => {})
    } catch (e) {
      console.error(e)
    } finally {
      setIsScaling(false)
    }
  }

  const handleRunCompression = async () => {
    setIsCompressing(true)
    try {
      const r = await runAxiomCompression()
      setCompressionResult(r)
    } catch (e) {
      console.error(e)
    } finally {
      setIsCompressing(false)
    }
  }

  // Calculate circular SVG progress percentages for KRONOS Gauges
  const calculateCircleDash = (value, max) => {
    const r = 28
    const circ = 2 * Math.PI * r
    const num = parseFloat(value) || 0
    const pct = Math.min((num / max) * 100, 100)
    return circ - (pct / 100) * circ
  }

  return (
    <div style={{ animation: 'fadeUp 0.4s ease', display: 'flex', gap: '24px', flexWrap: 'wrap', alignItems: 'stretch' }}>
      <div style={{ flex: '3 1 350px', minWidth: '350px' }}>
      {/* Sliding glow tab bar */}
      <div 
        className="glass-panel" 
        style={{ 
          display: 'inline-flex', 
          padding: 6, 
          gap: 6, 
          marginBottom: 28, 
          borderRadius: 10,
          border: '1px solid var(--border)' 
        }}
      >
        <button
          className={`btn ${activeTab === 'predictions' ? 'btn-primary' : ''}`}
          onClick={() => setActiveTab('predictions')}
          style={{ 
            borderRadius: 6, 
            fontSize: 13, 
            background: activeTab === 'predictions' ? undefined : 'transparent',
            border: 'none',
            color: activeTab === 'predictions' ? '#040712' : 'var(--text-secondary)',
            display: 'flex',
            alignItems: 'center',
            gap: 7,
          }}
        >
          ◎ Causal Predictions
          {predictions.length > 0 && (
            <span style={{ fontSize: 9, fontWeight: 800, background: 'rgba(0,255,255,0.15)', color: 'var(--cyan)', border: '1px solid rgba(0,255,255,0.25)', borderRadius: 10, padding: '1px 6px', lineHeight: '1.4' }}>
              {predictions.length}
            </span>
          )}
        </button>
        <button
          className={`btn ${activeTab === 'kronos' ? 'btn-primary' : ''}`}
          onClick={() => setActiveTab('kronos')}
          style={{ 
            borderRadius: 6, 
            fontSize: 13, 
            background: activeTab === 'kronos' ? undefined : 'transparent',
            border: 'none',
            color: activeTab === 'kronos' ? '#040712' : 'var(--text-secondary)',
            display: 'flex',
            alignItems: 'center',
            gap: 7,
          }}
        >
          ✦ KRONOS Control Room
          {optimizationHistory.length > 0 && (
            <span style={{ fontSize: 9, fontWeight: 800, background: 'rgba(0,255,255,0.15)', color: activeTab === 'kronos' ? '#040712' : 'var(--cyan)', border: '1px solid rgba(0,255,255,0.25)', borderRadius: 10, padding: '1px 6px', lineHeight: '1.4' }}>
              {optimizationHistory.length}
            </span>
          )}
        </button>
      </div>

      {activeTab === 'predictions' ? (
        <div>
          {/* Brief count */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 20, marginBottom: 28 }}>
            <GlassCard glowType="cyan">
              <div className="stat-label">AI Intelligence Briefs</div>
              <div className="stat-value mono">
                <AnimatedCounter value={predictions.length} />
              </div>
              <div className="stat-sub">Active ZOLA causal paths computed</div>
            </GlassCard>
            <GlassCard glowType={status?.entity_mode === 'live' ? 'cyan' : 'amber'}>
              <div className="stat-label">Prediction Resolution Mode</div>
              <div className="stat-value mono" style={{ fontSize: 24, textTransform: 'uppercase', color: status?.entity_mode === 'live' ? 'var(--cyan)' : 'var(--amber)' }}>
                {status?.entity_mode ?? 'MOCK'}
              </div>
              <div className="stat-sub">PyTorch model dynamic parameters</div>
            </GlassCard>
          </div>

          {predictions.length === 0 ? (
            <GlassCard style={{ textAlign: 'center', padding: '60px 20px', color: 'var(--text-muted)' }}>
              No active pre-transition behavior signatures. Inject data events or wait for entropy spikes!
            </GlassCard>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
              {predictions.map((p, i) => (
                <GlassCard 
                  key={i} 
                  glowType="cyan" 
                  style={{ 
                    borderLeft: '4px solid var(--cyan)',
                    padding: '24px 28px' 
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
                    <div>
                      <h3 style={{ fontSize: 18, fontWeight: 700 }}>{p.legal_name || p.entity_name}</h3>
                      <div className="mono" style={{ fontSize: 11, color: 'var(--cyan)', marginTop: 4, letterSpacing: '0.8px' }}>
                        TRANSITION: {p.prediction_details?.transition_type?.replace(/_/g, ' ').toUpperCase() || 'EXPANSION ACTION'}
                      </div>
                    </div>
                    <div style={{ textAlign: 'right' }}>
                      <div className="mono" style={{ fontSize: 24, fontWeight: 800, color: 'var(--cyan)' }}>
                        {`${Math.round((p.expansion_score ?? p.prediction_details?.success_probability ?? 0) * 100)}%`}
                      </div>
                      <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                        Expansion Score
                      </div>
                    </div>
                  </div>

                  {/* Horizontal visual progress gauge */}
                  <div style={{ width: '100%', height: 4, background: 'rgba(255,255,255,0.05)', borderRadius: 2, overflow: 'hidden', marginBottom: 20 }}>
                    <div 
                      style={{ 
                        height: '100%', 
                        width: `${(p.expansion_score ?? p.prediction_details?.success_probability ?? 0) * 100}%`, 
                        background: 'linear-gradient(90deg, var(--blue), var(--cyan))',
                        boxShadow: '0 0 8px var(--cyan)'
                      }} 
                    />
                  </div>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: 10, fontSize: 13, marginBottom: 20 }}>
                    {p.narrative && (
                      <div style={{ marginBottom: 6 }}>
                        <span style={{ color: 'var(--text-muted)', fontWeight: 'bold' }}>GENERATED NARRATIVE:</span>
                        <p style={{ color: 'var(--text-secondary)', margin: '4px 0 0 0', fontStyle: 'italic', lineHeight: '1.4' }}>{p.narrative}</p>
                      </div>
                    )}
                    <div>
                      <span style={{ color: 'var(--text-muted)', fontWeight: 'bold' }}>CAUSAL MECHANISM:</span>
                      <span style={{ color: 'var(--text-secondary)', marginLeft: 8 }}>{p.prediction_details?.causal_mechanism || p.causal_mechanism}</span>
                    </div>
                    <div>
                      <span style={{ color: 'var(--text-muted)', fontWeight: 'bold' }}>OPTIMAL INTERVENTION:</span>
                      <span style={{ color: 'var(--cyan)', fontWeight: 600, marginLeft: 8 }}>{p.prediction_details?.optimal_intervention || p.optimal_intervention}</span>
                    </div>
                    <div>
                      <span style={{ color: 'var(--text-muted)', fontWeight: 'bold' }}>RECOMMENDED TIMING:</span>
                      <span style={{ color: 'var(--text-secondary)', marginLeft: 8 }}>{p.prediction_details?.recommended_timing || p.recommended_timing}</span>
                    </div>
                  </div>


                  {/* Consequence chain flowchart */}
                  {p.consequence_chain?.length > 0 && (
                    <div style={{ borderTop: '1px solid rgba(255,255,255,0.04)', paddingTop: 16 }}>
                      <div style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 'bold', letterSpacing: '1px', marginBottom: 12 }}>
                        COMPUTED CONSEQUENCE CHAIN
                      </div>
                      
                      <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }} className="mono">
                        {p.consequence_chain.map((c, j) => (
                          <div key={j} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                            <div 
                              style={{ 
                                padding: '6px 12px', 
                                background: 'rgba(255,255,255,0.02)', 
                                border: '1px solid var(--border)', 
                                borderRadius: 6,
                                fontSize: 11,
                                color: 'var(--text-secondary)'
                              }}
                            >
                              {c}
                            </div>
                            {j < p.consequence_chain.length - 1 && (
                              <span style={{ color: 'var(--cyan)', fontWeight: 'bold', animation: 'pulse 1.5s infinite' }}>
                                ➜
                              </span>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </GlassCard>
              ))}
            </div>
          )}
        </div>
      ) : (
        /* KRONOS & Self-Evolution Dashboard */
        <div ref={kronosRef}>

          {/* ── KRONOS Status Bar ── */}
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            padding: '8px 16px',
            background: 'rgba(0,255,255,0.05)',
            border: '1px solid rgba(0,255,255,0.1)',
            borderRadius: 6,
            marginBottom: 16,
            flexWrap: 'wrap',
            gap: 8,
          }}>
            <div style={{ display: 'flex', gap: 14, alignItems: 'center', flexWrap: 'wrap' }}>
              {/* Pill: Entity mode */}
              <span className="mono" style={{
                fontSize: 10, fontWeight: 700,
                color: status?.entity_mode === 'live' ? 'var(--cyan)' : 'rgba(255,255,255,0.3)',
              }}>
                ● ENTITY MODE: {(status?.entity_mode ?? 'MOCK').toUpperCase()}
              </span>
              {/* Pill: CIFN active */}
              <span className="mono" style={{
                fontSize: 10, fontWeight: 700,
                color: optimizationHistory.length > 0 ? '#00ff88' : 'rgba(255,255,255,0.3)',
              }}>
                ● CIFN {optimizationHistory.length > 0 ? 'ACTIVE' : 'STANDBY'}
              </span>
              {/* Pill: Evolution ready */}
              <span className="mono" style={{
                fontSize: 10, fontWeight: 700,
                color: status?.stats?.pending_patches?.length > 0 ? '#00ff88' : 'rgba(255,255,255,0.3)',
              }}>
                ● EVOLUTION {status?.stats?.pending_patches?.length > 0 ? 'READY' : 'IDLE'}
              </span>
            </div>
            <span className="mono" style={{ fontSize: 11, color: 'var(--text-muted)', opacity: 0.6 }}>
              KRONOS v1.0 · Continuous Interference Field Network
            </span>
          </div>

          {/* ── LIVE Model Intelligence Panel ── */}
          <div style={{ marginBottom: 24 }}>
            <div className="mono" style={{
              fontSize: 11, fontWeight: 800, letterSpacing: '1.5px',
              color: 'var(--cyan)', textTransform: 'uppercase', marginBottom: 12,
              display: 'flex', alignItems: 'center', gap: 8
            }}>
              <span style={{ opacity: 0.5 }}>◈</span>
              Live KRONOS Model Intelligence
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 12 }}>
              {[
                {
                  label: 'KRONOS PILLARS',
                  value: architectureReport?.kronos?.pillars ?? '—',
                  sub: architectureReport?.kronos?.pillar_names?.slice(0,3).join(', ') + (architectureReport?.kronos?.pillars > 3 ? '…' : '') || 'Loading…',
                  color: 'var(--cyan)'
                },
                {
                  label: 'LIVE PARAMS',
                  value: (architectureReport?.kronos?.params ?? 0).toLocaleString(),
                  sub: `${((architectureReport?.kronos?.params ?? 0) / 1e6).toFixed(2)}M active parameters`,
                  color: 'var(--cyan)'
                },
                {
                  label: 'APEX MORPHISMS',
                  value: architectureReport?.apex?.n_total_morphisms ?? '—',
                  sub: `Causal depth: ${architectureReport?.apex?.max_causal_depth ?? '—'}`,
                  color: 'var(--blue)'
                },
                {
                  label: 'DRSN NODES',
                  value: architectureReport?.drsn?.n_nodes ?? '—',
                  sub: `Mean V: ${architectureReport?.drsn?.mean_V?.toFixed(1) ?? '—'}mV`,
                  color: 'var(--blue)'
                },
                {
                  label: 'GÖDEL FITNESS',
                  value: scalingStatus?.fitness_history?.length > 0
                    ? scalingStatus.fitness_history[scalingStatus.fitness_history.length - 1].toFixed(4)
                    : '—',
                  sub: `Generations: ${scalingStatus?.generations_completed ?? 0}`,
                  color: '#00ff88'
                },
                {
                  label: 'BACKPROP STEPS',
                  value: (status?.stats?.backprop_steps ?? 0).toLocaleString(),
                  sub: `Loss: ${status?.stats?.latest_loss?.toFixed(4) ?? '—'}`,
                  color: 'var(--amber)'
                },
              ].map(({ label, value, sub, color }) => (
                <div key={label} style={{
                  background: 'rgba(0,0,0,0.3)',
                  border: '1px solid rgba(0,255,255,0.15)',
                  borderRadius: 6, padding: 12,
                }}>
                  <div className="mono" style={{ fontSize: 9, color: 'var(--text-muted)', fontWeight: 800, letterSpacing: '1px', textTransform: 'uppercase', marginBottom: 6 }}>
                    {label}
                  </div>
                  <div className="mono" style={{ fontSize: 17, fontWeight: 700, color, marginBottom: 3 }}>
                    {value}
                  </div>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>{sub}</div>
                </div>
              ))}
            </div>

            {/* Pillar badges */}
            {architectureReport?.kronos?.pillar_names?.length > 0 && (
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 12 }}>
                {architectureReport.kronos.pillar_names.map(name => (
                  <span key={name} style={{
                    fontSize: 9, padding: '2px 8px',
                    border: '1px solid rgba(0,255,255,0.25)', borderRadius: 10,
                    color: 'var(--cyan)', fontFamily: 'monospace'
                  }}>{name}</span>
                ))}
              </div>
            )}
          </div>


          {/* ── Architecture Report Panel ── */}
          <div style={{ marginBottom: 24 }}>
            <div className="mono" style={{
              fontSize: 11,
              fontWeight: 800,
              letterSpacing: '1.5px',
              color: 'var(--cyan)',
              textTransform: 'uppercase',
              marginBottom: 12,
              display: 'flex',
              alignItems: 'center',
              gap: 8
            }}>
              <span style={{ opacity: 0.5 }}>◈</span>
              Full Architecture Report
            </div>
            <GlassCard glowType="cyan">
              {(!architectureReport || architectureReport.available === false) ? (
                <div style={{ color: 'var(--text-muted)', fontSize: 12 }}>
                  Architecture report unavailable in mock mode
                </div>
              ) : (
                <>
                  {/* 4-column stat grid */}
                  <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 16 }}>
                    {[
                      {
                        label: 'DRSN SYNAPTIC LAYER',
                        value: (architectureReport.drsn?.n_nodes ?? '—') + ' nodes',
                        sub: 'mean V: ' + (architectureReport.drsn?.mean_V?.toFixed(2) ?? '—') + 'mV',
                      },
                      {
                        label: architectureReport.kronos?.model === 'NOETHER_KRONOS' ? 'NOETHER-KRONOS 13-PILLAR' : 'KRONOS 9-PILLAR',
                        value: (architectureReport.kronos?.params?.toLocaleString() ?? '—') + ' params',
                        sub: 'pillars: ' + (architectureReport.kronos?.pillars ?? '—'),
                      },
                      {
                        label: 'CSIE SHEAF LAYER',
                        value: (architectureReport.sheaf?.n_coverings ?? '—') + ' coverings',
                        sub: (architectureReport.sheaf?.n_total_sections ?? '—') + ' sections',
                      },
                      {
                        label: 'APEX CAUSAL ENGINE',
                        value: (architectureReport.apex?.n_total_morphisms ?? '—') + ' morphisms',
                        sub: 'depth: ' + (architectureReport.apex?.max_causal_depth ?? '—'),
                      },
                    ].map(({ label, value, sub }) => (
                      <div key={label} style={{
                        background: 'rgba(0,0,0,0.3)',
                        border: '1px solid rgba(0,255,255,0.15)',
                        borderRadius: 6,
                        padding: 12,
                        minWidth: 140,
                      }}>
                        <div className="mono" style={{ fontSize: 9, color: 'var(--text-muted)', fontWeight: 800, letterSpacing: '1px', textTransform: 'uppercase', marginBottom: 6 }}>
                          {label}
                        </div>
                        <div className="mono" style={{ fontSize: 15, fontWeight: 700, color: 'var(--cyan)', marginBottom: 3 }}>
                          {value}
                        </div>
                        <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                          {sub}
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Architecture layer badges */}
                  {architectureReport.stats?.architecture_layers?.length > 0 && (
                    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                      {architectureReport.stats.architecture_layers.map((layer) => (
                        <span key={layer} style={{
                          fontSize: 10,
                          padding: '3px 8px',
                          border: '1px solid rgba(0,255,255,0.3)',
                          borderRadius: 10,
                          color: 'var(--cyan)',
                        }}>
                          {layer}
                        </span>
                      ))}
                    </div>
                  )}
                </>
              )}
            </GlassCard>
          </div>

          {/* ── AXIOM Zero-Loss Compression Analysis ── */}
          <div style={{ marginBottom: 24 }}>
            <div className="mono" style={{
              fontSize: 11,
              fontWeight: 800,
              letterSpacing: '1.5px',
              color: 'var(--cyan)',
              textTransform: 'uppercase',
              marginBottom: 12,
              display: 'flex',
              alignItems: 'center',
              gap: 8
            }}>
              <span style={{ opacity: 0.5 }}>◈</span>
              AXIOM Zero-Loss Compression Analysis
            </div>
            <GlassCard glowType="cyan">
              {(!axiomReport || axiomReport.available === false) ? (
                <div style={{ color: 'var(--text-muted)', fontSize: 12 }}>
                  Compression analysis unavailable
                </div>
              ) : (
                <>
                  {/* Top stat row */}
                  <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 16 }}>
                    {[
                      { label: 'TOTAL PARAMS',  value: axiomReport.total_params?.toLocaleString() ?? '—' },
                      { label: 'EST. SIZE',      value: (axiomReport.estimated_total_size_mb?.toFixed(2) ?? '—') + ' MB' },
                      { label: 'COMPRESSED',     value: (axiomReport.estimated_compressed_mb?.toFixed(2) ?? '—') + ' MB' },
                      { label: 'MEAN ENTROPY',   value: (axiomReport.mean_entropy_bits?.toFixed(2) ?? '—') + ' bits' },
                    ].map(({ label, value }) => (
                      <div key={label} style={{
                        background: 'rgba(0,0,0,0.3)',
                        border: '1px solid rgba(0,255,255,0.15)',
                        borderRadius: 6,
                        padding: 12,
                        minWidth: 140,
                      }}>
                        <div className="mono" style={{ fontSize: 9, color: 'var(--text-muted)', fontWeight: 800, letterSpacing: '1px', textTransform: 'uppercase', marginBottom: 6 }}>
                          {label}
                        </div>
                        <div className="mono" style={{ fontSize: 15, fontWeight: 700, color: 'var(--cyan)' }}>
                          {value}
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Per-layer table */}
                  {axiomReport.layer_reports?.length > 0 && (
                    <div style={{ maxHeight: 200, overflowY: 'auto', marginBottom: 16, borderRadius: 6, border: '1px solid rgba(0,255,255,0.1)' }}>
                      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                        <thead>
                          <tr style={{ background: 'rgba(0,255,255,0.07)' }}>
                            {['Layer', 'Params', 'Entropy', 'Compression', 'Null Space'].map(col => (
                              <th key={col} className="mono" style={{ padding: '6px 10px', textAlign: 'left', fontSize: 9, color: 'var(--text-muted)', fontWeight: 800, letterSpacing: '0.8px', textTransform: 'uppercase', whiteSpace: 'nowrap' }}>
                                {col}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {axiomReport.layer_reports.map((row, i) => (
                            <tr key={i} style={{ background: i % 2 === 0 ? 'rgba(0,255,255,0.03)' : 'transparent' }}>
                              <td className="mono" style={{ padding: '5px 10px', color: 'var(--text-secondary)', maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{row.name}</td>
                              <td className="mono" style={{ padding: '5px 10px', color: 'var(--text-secondary)' }}>{row.n_params?.toLocaleString()}</td>
                              <td className="mono" style={{ padding: '5px 10px', color: 'var(--cyan)' }}>{row.entropy_bits?.toFixed(2)}</td>
                              <td className="mono" style={{ padding: '5px 10px', color: 'var(--text-secondary)' }}>{row.estimated_compression_ratio?.toFixed(3)}</td>
                              <td className="mono" style={{ padding: '5px 10px', color: 'var(--text-secondary)' }}>{(row.null_space_estimate * 100)?.toFixed(1)}%</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}

                  {/* Phases available as green pills */}
                  {axiomReport.phases_available?.length > 0 && (
                    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                      {axiomReport.phases_available.map((phase) => (
                        <span key={phase} style={{
                          fontSize: 10,
                          padding: '3px 8px',
                          border: '1px solid rgba(0,255,136,0.3)',
                          borderRadius: 10,
                          color: '#00ff88',
                        }}>
                          {phase}
                        </span>
                      ))}
                    </div>
                  )}
                </>
              )}
            </GlassCard>
          </div>

          {/* ── CIFN Architecture Panel ── */}
          <div style={{ marginBottom: 24 }}>
            <div className="mono" style={{
              fontSize: 11,
              fontWeight: 800,
              letterSpacing: '1.5px',
              color: 'var(--cyan)',
              textTransform: 'uppercase',
              marginBottom: 12,
              display: 'flex',
              alignItems: 'center',
              gap: 8
            }}>
              <span style={{ opacity: 0.5 }}>◈</span>
              Continuous Interference Field Network (CIFN)
            </div>

            <GlassCard glowType="cyan">
              <div style={{ display: 'flex', gap: 32, flexWrap: 'wrap' }}>

                {/* Left column — How it works */}
                <div style={{ flex: '2 1 260px' }}>
                  <div className="mono" style={{ fontSize: 9, color: 'var(--text-muted)', fontWeight: 800, letterSpacing: '1px', textTransform: 'uppercase', marginBottom: 10 }}>
                    How it works
                  </div>
                  <p style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: '1.6', marginBottom: 10 }}>
                    Traditional networks store every weight value explicitly. A 1000×1000 layer requires 1,000,000 stored floats.
                  </p>
                  <p style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: '1.6', marginBottom: 10 }}>
                    CIFN replaces stored weights with a continuous sinusoidal basis. The weight matrix W is computed on every forward pass as:
                  </p>
                  <pre className="mono" style={{
                    background: 'rgba(0,0,0,0.4)',
                    border: '1px solid rgba(0,255,255,0.12)',
                    borderRadius: 8,
                    padding: '10px 14px',
                    fontSize: 12,
                    color: '#00ffff',
                    letterSpacing: '0.3px',
                    lineHeight: '1.7',
                    marginBottom: 10,
                    overflowX: 'auto',
                    textShadow: '0 0 8px rgba(0,255,255,0.3)',
                    whiteSpace: 'pre-wrap'
                  }}>
                    {`W[i,j] = \u03a3\u2096 a\u2096 \u00b7 sin(\u03c9_out\u2096 \u00b7 x\u1d62 + \u03b8_out\u2096) \u00b7 sin(\u03c9_in\u2096 \u00b7 y\u2c7c + \u03b8_in\u2096)`}
                  </pre>
                  <p style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: '1.6' }}>
                    {`Only the wave parameters (amplitudes a\u2096, frequencies \u03c9, phases \u03b8) are stored. The full weight matrix is generated on-the-fly.`}
                  </p>
                </div>

                {/* Right column — Architecture */}
                <div style={{ flex: '1 1 180px', display: 'flex', flexDirection: 'column', gap: 8 }}>
                  <div className="mono" style={{ fontSize: 9, color: 'var(--text-muted)', fontWeight: 800, letterSpacing: '1px', textTransform: 'uppercase', marginBottom: 2 }}>
                    Architecture
                  </div>

                  {[
                    { label: 'Input Layer',  value: '8 features' },
                    { label: 'Hidden Layer', value: '16 neurons \u00b7 basis=128' },
                    { label: 'Output Layer', value: '15 classes \u00b7 basis=128' },
                  ].map(({ label, value }) => (
                    <div key={label} style={{
                      background: 'rgba(255,255,255,0.02)',
                      border: '1px solid var(--border)',
                      borderRadius: 6,
                      padding: '8px 12px',
                    }}>
                      <div style={{ fontSize: 9, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.8px', marginBottom: 2 }}>{label}</div>
                      <div className="mono" style={{ fontSize: 12, color: 'var(--text-primary)', fontWeight: 600 }}>{value}</div>
                    </div>
                  ))}

                  <div style={{ marginTop: 4, display: 'flex', flexDirection: 'column', gap: 6 }}>
                    <div style={{
                      background: 'rgba(0,255,255,0.04)',
                      border: '1px solid rgba(0,255,255,0.15)',
                      borderRadius: 6,
                      padding: '8px 12px',
                    }}>
                      <div style={{ fontSize: 9, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.8px', marginBottom: 2 }}>Basis Parameters</div>
                      <div className="mono" style={{ fontSize: 12, color: 'var(--cyan)', fontWeight: 700 }}>
                        {optimizationHistory.length > 0
                          ? (optimizationHistory[optimizationHistory.length - 1].architecture?.actual_trainable_params?.toLocaleString() ?? '~2,048') + ' stored'
                          : '~2,048 stored'
                        }
                      </div>
                    </div>
                    <div style={{
                      background: 'rgba(255,255,255,0.02)',
                      border: '1px solid var(--border)',
                      borderRadius: 6,
                      padding: '8px 12px',
                    }}>
                      <div style={{ fontSize: 9, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.8px', marginBottom: 2 }}>Weight Representation</div>
                      <div className="mono" style={{ fontSize: 11, color: 'var(--text-secondary)' }}>Computed on forward pass</div>
                    </div>
                  </div>
                </div>

              </div>
            </GlassCard>
          </div>

          {/* ── Divider ── */}
          <div style={{ height: 1, background: 'linear-gradient(to right, transparent, rgba(0,255,255,0.3), transparent)', margin: '20px 0' }} />

          {/* Main system stats and circular gauges */}
          <div className="grid-2" style={{ marginBottom: 24 }}>
            {/* Multi-Phase parameters metrics */}
            <GlassCard title="KRONOS Parameter Scaling Registers" glowType="blue">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 10 }}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 10, flex: 1 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
                    <span style={{ color: 'var(--text-muted)' }}>Virtual Parameters:</span>
                    <span className="mono" style={{ color: 'var(--blue)', fontWeight: 'bold' }}>
                      {status?.stats?.virtual_parameters?.toLocaleString() ?? "N/A (Mock Mode)"} (Theoretical 1Q Target)
                    </span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
                    <span style={{ color: 'var(--text-muted)' }}>Wave Basis Storage:</span>
                    <span className="mono" style={{ color: 'var(--cyan)' }}>
                      {status?.stats?.wave_basis_size_kb ? `${status.stats.wave_basis_size_kb} KB` : "0.0 KB"}
                    </span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
                    <span style={{ color: 'var(--text-muted)' }}>Backpropagation Steps:</span>
                    <span className="mono">{status?.stats?.backprop_steps ?? 0}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
                    <span style={{ color: 'var(--text-muted)' }}>Self-Evolution Cycles:</span>
                    <span className="mono">{status?.stats?.self_evolution_cycles ?? 0}</span>
                  </div>
                </div>

                {/* SVG Dial Gauge parameter scale percentage */}
                <div style={{ position: 'relative', width: 90, height: 90, display: 'flex', alignItems: 'center', justifyContent: 'center', marginLeft: 24 }}>
                  <svg width="90" height="90" style={{ transform: 'rotate(-90deg)' }}>
                    <circle cx="45" cy="45" r="28" fill="transparent" stroke="rgba(255,255,255,0.03)" strokeWidth="5" />
                    <circle 
                      cx="45" 
                      cy="45" 
                      r="28" 
                      fill="transparent" 
                      stroke="var(--blue)" 
                      strokeWidth="5" 
                      strokeDasharray={2 * Math.PI * 28}
                      strokeDashoffset={calculateCircleDash(status?.stats?.virtual_parameters ?? 0, 1e15)}
                      style={{ transition: 'stroke-dashoffset 0.8s ease' }}
                    />
                  </svg>
                  <div style={{ position: 'absolute', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                    <span className="mono" style={{ fontSize: 10, fontWeight: 'bold', color: 'var(--blue)' }}>
                      {status?.stats?.virtual_parameters 
                        ? `${(status.stats.virtual_parameters / 1e15 * 100).toFixed(4)}%` 
                        : '0.00%'}
                    </span>
                    <span style={{ fontSize: 7, color: 'var(--text-muted)' }}>1Q SCALE</span>
                  </div>
                </div>
              </div>
            </GlassCard>

            {/* Gradient Flow Optimizers */}
            <GlassCard title="Gradient Flow Lossless Metrics" glowType="cyan">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 10 }}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 10, flex: 1 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
                    <span style={{ color: 'var(--text-muted)' }}>Latest PyTorch Loss:</span>
                    <span className="mono" style={{ color: 'var(--magenta)', fontWeight: 'bold' }}>
                      {status?.stats?.latest_loss ?? "0.0000"}
                    </span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
                    <span style={{ color: 'var(--text-muted)' }}>Gradient Norm (a, ω, θ):</span>
                    <span className="mono" style={{ color: 'var(--cyan)' }}>
                      {status?.stats?.latest_grad_norm ?? "0.0000"}
                    </span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
                    <span style={{ color: 'var(--text-muted)' }}>Cyberspace Facts Learned:</span>
                    <span className="mono">{status?.stats?.facts_crawled ?? 0}</span>
                  </div>
                </div>

                {/* Cyber actions trigger buttons */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginLeft: 20 }}>
                  <button
                    className="btn btn-primary"
                    disabled={loading || status?.entity_mode !== 'live'}
                    onClick={handleLearn}
                    style={{ padding: '8px 12px', fontSize: 11 }}
                  >
                    {loading ? 'Crawling...' : 'Crawl Cyberspace'}
                  </button>
                  <button
                    className="btn"
                    disabled={loading || status?.entity_mode !== 'live'}
                    onClick={handlePropose}
                    style={{ padding: '8px 12px', fontSize: 11, background: 'transparent', border: '1px solid var(--border)' }}
                  >
                    Propose Hotpatch
                  </button>
                </div>
              </div>
            </GlassCard>
          </div>

          {/* Autonomous Self-Evolution Pipeline */}
          {(() => {
            const patch      = status?.stats?.pending_patches?.[0] ?? null
            const approved   = status?.stats?.approved_patches?.length > 0
            const patchId    = patch?.patch_id ?? null
            const patchCode  = patch?.patch_code ?? null
            const patchStatus = patch?.status ?? null

            // Derive step state
            // step 1: propose — complete when patch exists
            // step 2: validate — complete when status is sandbox_verified
            // step 3: approve — complete when approved_patches has entries
            const step1Done = !!patch
            const step2Done = patchStatus === 'sandbox_verified'
            const step3Done = approved

            const activeStep = step3Done ? 3 : step2Done ? 3 : step1Done ? 2 : 1

            const STEPS = [
              { n: 1, label: 'PROPOSE',  sub: 'Generate patch code' },
              { n: 2, label: 'VALIDATE', sub: 'Sandbox compile check' },
              { n: 3, label: 'APPROVE',  sub: 'Apply to live model' },
            ]

            const stepColor = (n) => {
              const done = (n === 1 && step1Done) || (n === 2 && step2Done) || (n === 3 && step3Done)
              if (done) return { border: '1px solid #00ff88', color: '#00ff88' }
              if (n === activeStep) return { border: '1px solid var(--cyan)', color: 'var(--cyan)' }
              return { border: '1px solid rgba(255,255,255,0.1)', color: 'rgba(255,255,255,0.3)' }
            }

            const badge = (n) => {
              if (n === 1) {
                if (step1Done) return { label: 'COMPLETE', color: '#00ff88' }
                if (loading && activeStep === 1) return { label: 'IN PROGRESS', color: '#ffd93d', pulse: true }
                return { label: 'PENDING', color: 'rgba(255,255,255,0.3)' }
              }
              if (n === 2) {
                if (step2Done) return { label: 'COMPLETE', color: '#00ff88' }
                if (loading && activeStep === 2) return { label: 'IN PROGRESS', color: '#ffd93d', pulse: true }
                return { label: 'PENDING', color: 'rgba(255,255,255,0.3)' }
              }
              if (step3Done) return { label: 'COMPLETE', color: '#00ff88' }
              if (loading && activeStep === 3) return { label: 'IN PROGRESS', color: '#ffd93d', pulse: true }
              return { label: 'PENDING', color: 'rgba(255,255,255,0.3)' }
            }

            return (
              <div style={{ marginBottom: 24 }}>
                <GlassCard glowType="cyan">

                  {/* Card header */}
                  <div className="mono" style={{ fontSize: 12, fontWeight: 800, letterSpacing: '1.2px', color: 'var(--cyan)', textTransform: 'uppercase', marginBottom: 18, display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span style={{ fontSize: 16 }}>⟳</span> Autonomous Self-Evolution Pipeline
                  </div>

                  {/* Horizontal stepper */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
                    {STEPS.map((s, idx) => {
                      const sc = stepColor(s.n)
                      return (
                        <div key={s.n} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          <div style={{ padding: '10px 18px', borderRadius: 8, ...sc, textAlign: 'center', minWidth: 110 }}>
                            <div className="mono" style={{ fontSize: 11, fontWeight: 800, letterSpacing: '1px' }}>
                              {s.n === activeStep && !( (s.n===1&&step1Done)||(s.n===2&&step2Done)||(s.n===3&&step3Done) ) ? '▶ ' : ''}{s.label}
                            </div>
                            <div style={{ fontSize: 9, opacity: 0.7, marginTop: 2 }}>{s.sub}</div>
                          </div>
                          {idx < STEPS.length - 1 && (
                            <span style={{ color: 'rgba(255,255,255,0.2)', fontSize: 18, fontWeight: 300 }}>→</span>
                          )}
                        </div>
                      )
                    })}
                  </div>

                  {/* Two-column body */}
                  <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>

                    {/* LEFT — Pipeline controls */}
                    <div style={{ flex: 1, minWidth: 200, display: 'flex', flexDirection: 'column', gap: 10 }}>
                      <div className="mono" style={{ fontSize: 9, color: 'var(--text-muted)', fontWeight: 800, letterSpacing: '1px', textTransform: 'uppercase', marginBottom: 2 }}>Pipeline Controls</div>

                      {[
                        {
                          icon: '①',
                          label: 'PROPOSE PATCH',
                          onClick: handlePropose,
                          disabled: loading || status?.entity_mode !== 'live',
                          badge: badge(1),
                        },
                        {
                          icon: '②',
                          label: 'VALIDATE PATCH',
                          onClick: () => patchId && handleValidate(patchId),
                          disabled: loading || !patchId || step2Done,
                          badge: badge(2),
                        },
                        {
                          icon: '③',
                          label: 'APPROVE & APPLY',
                          onClick: () => patchId && handleApprove(patchId),
                          disabled: loading || !step2Done || step3Done,
                          badge: badge(3),
                        },
                      ].map((btn) => (
                        <button
                          key={btn.label}
                          className="btn"
                          onClick={btn.onClick}
                          disabled={btn.disabled}
                          style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'center',
                            padding: '10px 14px',
                            fontSize: 12,
                            background: 'rgba(255,255,255,0.03)',
                            border: '1px solid var(--border)',
                            borderRadius: 8,
                            cursor: btn.disabled ? 'not-allowed' : 'pointer',
                            opacity: btn.disabled ? 0.5 : 1,
                            width: '100%',
                            textAlign: 'left',
                          }}
                        >
                          <span className="mono">{btn.icon} {btn.label}</span>
                          <span
                            className="mono"
                            style={{
                              fontSize: 9,
                              fontWeight: 800,
                              color: btn.badge.color,
                              letterSpacing: '0.8px',
                              animation: btn.badge.pulse ? 'pulse 1s infinite' : 'none',
                            }}
                          >
                            {btn.badge.label}
                          </span>
                        </button>
                      ))}
                    </div>

                    {/* RIGHT — Patch code viewer */}
                    <div style={{ flex: 1, minWidth: 200 }}>
                      <div className="mono" style={{ fontSize: 9, color: 'var(--text-muted)', fontWeight: 800, letterSpacing: '1px', textTransform: 'uppercase', marginBottom: 6 }}>Patch Code</div>
                      <div style={{ position: 'relative' }}>
                        <pre
                          className="mono"
                          style={{
                            height: 200,
                            overflow: 'auto',
                            background: 'rgba(0,0,0,0.5)',
                            border: '1px solid rgba(0,255,136,0.15)',
                            borderRadius: 6,
                            padding: 12,
                            fontSize: 11,
                            color: '#00ff88',
                            lineHeight: '1.6',
                            margin: 0,
                            whiteSpace: 'pre-wrap',
                            wordBreak: 'break-all',
                          }}
                        >
                          {patchCode
                            ? patchCode
                            : <span style={{ color: 'rgba(255,255,255,0.2)' }}>{'// patch code will appear here after PROPOSE step'}</span>
                          }
                        </pre>
                        {step3Done && (
                          <div style={{
                            position: 'absolute',
                            inset: 0,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            background: 'rgba(0,0,0,0.6)',
                            borderRadius: 6,
                            backdropFilter: 'blur(2px)',
                          }}>
                            <div className="mono" style={{ fontSize: 14, fontWeight: 800, color: '#00ff88', letterSpacing: '2px', textShadow: '0 0 12px #00ff88' }}>
                              ✓ APPLIED TO LIVE MODEL
                            </div>
                          </div>
                        )}
                      </div>
                      {patch?.error && (
                        <div className="mono" style={{ color: 'var(--red)', fontSize: 11, marginTop: 8, padding: '6px 10px', borderRadius: 6, background: 'rgba(255,0,60,0.05)', border: '1px solid rgba(255,0,60,0.15)' }}>
                          ✗ {patch.error}
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Action log terminal */}
                  <div style={{ marginTop: 16, borderTop: '1px solid rgba(255,255,255,0.04)', paddingTop: 14 }}>
                    <div className="mono" style={{ fontSize: 9, color: 'var(--text-muted)', fontWeight: 800, letterSpacing: '1px', textTransform: 'uppercase', marginBottom: 6 }}>Pipeline Output</div>
                    <div
                      className="crt-terminal"
                      style={{ background: '#02040b', border: '1px solid var(--border)', borderRadius: 8, padding: 12, minHeight: 80 }}
                    >
                      <div className="crt-scanner" />
                      <div
                        className="mono"
                        style={{ fontSize: 11, lineHeight: '1.6', color: '#4bf5a0', maxHeight: 140, overflowY: 'auto', textShadow: '0 0 3px rgba(75,245,160,0.4)' }}
                      >
                        {actionLog.length === 0 ? (
                          <div style={{ color: 'var(--text-muted)' }}>[SYSTEM] Daemon idle. Awaiting user commands...</div>
                        ) : (
                          actionLog.map((log, index) => {
                            let color = '#4bf5a0'
                            if (log.includes('ERROR:')) color = 'var(--red)'
                            if (log.includes('SUCCESS:')) color = 'var(--cyan)'
                            if (log.includes('INITIATING:') || log.includes('PROPOSED:')) color = 'var(--amber)'
                            return <div key={index} style={{ color }}>{log}</div>
                          })
                        )}
                        <div ref={terminalEndRef} />
                      </div>
                    </div>
                  </div>

                </GlassCard>
              </div>
            )
          })()}

          {/* ── Divider ── */}
          <div style={{ height: 1, background: 'linear-gradient(to right, transparent, rgba(0,255,255,0.3), transparent)', margin: '20px 0' }} />

          {/* Live Optimization Terminal Card */}
          <GlassCard glowType="cyan">
            {/* Card header: title + button */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
              <div className="mono" style={{ fontSize: 12, fontWeight: 800, letterSpacing: '1.2px', color: 'var(--cyan)', textTransform: 'uppercase' }}>
                ⚡ Live Optimization Terminal
              </div>
              <button
                className="btn btn-primary"
                onClick={handleRunOptimization}
                disabled={isOptimizing || status?.entity_mode !== 'live'}
                style={{ fontSize: 11, padding: '7px 14px' }}
              >
                {isOptimizing ? 'Running...' : '⚡ Run Differentiable Optimization Step'}
              </button>
            </div>

            {/* Two-column body */}
            <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>

              {/* LEFT — Optimization log */}
              <div style={{ flex: 1, minWidth: 220 }}>
                <div style={{ fontSize: 9, color: 'var(--text-muted)', fontWeight: 800, letterSpacing: '1px', textTransform: 'uppercase', marginBottom: 6 }} className="mono">Step Log</div>
                <div
                  ref={optScrollRef}
                  style={{
                    height: 220,
                    overflowY: 'auto',
                    background: 'rgba(0,0,0,0.4)',
                    border: '1px solid rgba(0,255,255,0.15)',
                    borderRadius: 6,
                    padding: 12,
                    fontFamily: 'monospace',
                    fontSize: 12,
                    lineHeight: '1.7'
                  }}
                >
                  {optimizationHistory.length === 0 ? (
                    <div style={{ color: 'rgba(255,255,255,0.25)' }}>&gt; awaiting optimization run...</div>
                  ) : (
                    optimizationHistory.map((r, i) => {
                      const loss = r.latest_loss
                      const lineColor = loss > 0.5 ? '#ff6b6b' : loss > 0.2 ? '#ffd93d' : '#00ff88'
                      return (
                        <div key={i} style={{ color: lineColor }}>
                          {`[STEP ${r.backprop_steps}] loss=${loss.toFixed(6)} grad=${r.latest_grad_norm.toFixed(4)} latency=${r.latency_ms}ms`}
                        </div>
                      )
                    })
                  )}
                </div>
              </div>

              {/* RIGHT — Dual-line chart */}
              <div style={{ flex: 1, minWidth: 220 }}>
                <div style={{ fontSize: 9, color: 'var(--text-muted)', fontWeight: 800, letterSpacing: '1px', textTransform: 'uppercase', marginBottom: 6 }} className="mono">Loss &amp; Grad Norm</div>
                <ResponsiveContainer width="100%" height={220}>
                  <LineChart
                    data={optimizationHistory.map((r, i) => ({ step: i + 1, loss: r.latest_loss, grad: r.latest_grad_norm }))}
                    margin={{ top: 4, right: 16, left: -20, bottom: 0 }}
                  >
                    <XAxis
                      dataKey="step"
                      tick={{ fill: 'var(--text-muted)', fontSize: 10 }}
                      axisLine={{ stroke: 'var(--border)' }}
                      tickLine={false}
                    />
                    <YAxis
                      yAxisId="left"
                      tick={{ fill: 'var(--text-muted)', fontSize: 10 }}
                      axisLine={{ stroke: 'var(--border)' }}
                      tickLine={false}
                      width={48}
                    />
                    <YAxis
                      yAxisId="right"
                      orientation="right"
                      tick={{ fill: '#ffd93d', fontSize: 10 }}
                      axisLine={{ stroke: 'var(--border)' }}
                      tickLine={false}
                      width={40}
                    />
                    <Tooltip
                      contentStyle={{
                        background: '#030710',
                        border: '1px solid var(--border)',
                        borderRadius: 6,
                        fontSize: 11,
                        color: '#00ffff'
                      }}
                      formatter={(v, name) => [v.toFixed(6), name === 'loss' ? 'Loss' : 'Grad Norm']}
                      labelFormatter={(l) => `Step ${l}`}
                    />
                    <Legend
                      wrapperStyle={{ fontSize: 10, color: 'var(--text-muted)', paddingTop: 6 }}
                      formatter={(value) => value === 'loss' ? 'Loss' : 'Grad Norm'}
                    />
                    <Line yAxisId="left"  type="monotone" dataKey="loss" stroke="#00ffff" strokeWidth={2} dot={false} isAnimationActive={false} />
                    <Line yAxisId="right" type="monotone" dataKey="grad" stroke="#ffd93d" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Stat strip */}
            {(() => {
              const latest = optimizationHistory[optimizationHistory.length - 1]
              const dash = '—'
              return (
                <div style={{ display: 'flex', gap: 24, marginTop: 16, paddingTop: 14, borderTop: '1px solid rgba(255,255,255,0.04)', flexWrap: 'wrap' }}>
                  {[
                    { label: 'LATENCY',          value: latest ? `${latest.latency_ms}ms`                                          : dash, color: 'var(--cyan)' },
                    { label: 'TRAINABLE PARAMS',  value: latest ? latest.architecture?.actual_trainable_params?.toLocaleString()    : dash, color: 'var(--blue)' },
                    { label: 'STORAGE',           value: latest ? latest.architecture?.weight_field_representation                   : dash, color: 'var(--text-secondary)' },
                    { label: 'STEPS COMPLETED',   value: latest ? latest.backprop_steps                                             : dash, color: 'var(--cyan)' },
                  ].map(({ label, value, color }) => (
                    <div key={label}>
                      <div className="stat-label">{label}</div>
                      <div className="mono" style={{ fontSize: 13, color, marginTop: 2 }}>{value}</div>
                    </div>
                  ))}
                </div>
              )
            })()}
          </GlassCard>

          {/* ── Divider ── */}
          <div style={{ height: 1, background: 'linear-gradient(to right, transparent, rgba(0,255,255,0.3), transparent)', margin: '20px 0' }} />

          {/* ── Live Intelligence Engine — Scaling & Compression Control ── */}
          <div style={{ marginBottom: 24 }}>
            <GlassCard glowType="cyan">

              {/* Card header */}
              <div className="mono" style={{ fontSize: 12, fontWeight: 800, letterSpacing: '1.2px', color: 'var(--cyan)', textTransform: 'uppercase', marginBottom: 18, display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ fontSize: 16 }}>⟳</span> LIVE INTELLIGENCE ENGINE — SCALING &amp; COMPRESSION CONTROL
              </div>

              {/* SECTION 1 — Gödel Auto Status pills */}
              <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 20 }}>
                {[
                  { label: 'OPTIMIZE CALLS', value: godelStatus?.optimize_calls_total ?? '—' },
                  { label: 'NEXT TRIGGER IN', value: godelStatus != null ? `${godelStatus.next_trigger_in} calls` : '—' },
                  { label: 'GENERATIONS', value: godelStatus?.generations_completed ?? '—' },
                  { label: 'FITNESS TREND', value: godelStatus?.fitness_trend ?? '—' },
                ].map(({ label, value }) => (
                  <div key={label} style={{
                    background: 'rgba(0,0,0,0.35)',
                    border: '1px solid rgba(0,255,255,0.2)',
                    borderRadius: 8,
                    padding: '10px 16px',
                    minWidth: 130,
                    flex: '1 1 120px',
                  }}>
                    <div className="mono" style={{ fontSize: 9, color: 'var(--text-muted)', fontWeight: 800, letterSpacing: '1px', textTransform: 'uppercase', marginBottom: 5 }}>
                      {label}
                    </div>
                    <div className="mono" style={{ fontSize: 15, fontWeight: 700, color: 'var(--cyan)' }}>
                      {value}
                    </div>
                  </div>
                ))}
              </div>

              {/* SECTION 2 — Action buttons */}
              <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
                <button
                  className="btn btn-primary"
                  onClick={handleRunScaling}
                  disabled={isScaling}
                  style={{ flex: '1 1 220px', padding: '11px 16px', fontSize: 12, fontWeight: 700 }}
                >
                  {isScaling ? 'Running...' : '▶ RUN GÖDEL SCALING GENERATION'}
                </button>
                <button
                  className="btn"
                  onClick={handleRunCompression}
                  disabled={isCompressing}
                  style={{
                    flex: '1 1 220px',
                    padding: '11px 16px',
                    fontSize: 12,
                    fontWeight: 700,
                    background: 'rgba(0,255,255,0.05)',
                    border: '1px solid rgba(0,255,255,0.3)',
                    color: 'var(--cyan)',
                    opacity: isCompressing ? 0.6 : 1,
                    cursor: isCompressing ? 'not-allowed' : 'pointer',
                  }}
                >
                  {isCompressing ? 'Compressing...' : '◈ APPLY AXIOM GAUGE COMPRESSION'}
                </button>
              </div>

              {/* SECTION 3 — Results */}
              {(scalingResult || compressionResult) && (
                <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>

                  {/* Scaling result */}
                  {scalingResult && (
                    <div style={{
                      flex: '1 1 240px',
                      background: 'rgba(0,0,0,0.3)',
                      border: '1px solid rgba(0,255,255,0.15)',
                      borderRadius: 8,
                      padding: 14,
                    }}>
                      <div className="mono" style={{ fontSize: 9, color: 'var(--text-muted)', fontWeight: 800, letterSpacing: '1px', textTransform: 'uppercase', marginBottom: 10 }}>
                        Gödel Scaling Result
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: 5, fontSize: 12 }}>
                        <div><span style={{ color: 'var(--text-muted)' }}>Generation: </span><span className="mono" style={{ color: 'var(--cyan)' }}>{scalingResult.generation}</span></div>
                        <div><span style={{ color: 'var(--text-muted)' }}>Best Fitness: </span><span className="mono" style={{ color: '#00ff88' }}>{scalingResult.best_fitness?.toFixed(4)}</span></div>
                        <div><span style={{ color: 'var(--text-muted)' }}>Fitness Trend: </span><span className="mono" style={{ color: 'var(--cyan)' }}>{scalingResult.fitness_trend}</span></div>
                      </div>
                      {scalingResult.best_config && Object.keys(scalingResult.best_config).length > 0 && (
                        <div style={{ marginTop: 10 }}>
                          <div className="mono" style={{ fontSize: 9, color: 'var(--text-muted)', fontWeight: 800, letterSpacing: '1px', textTransform: 'uppercase', marginBottom: 6 }}>
                            Best Config
                          </div>
                          <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap' }}>
                            {Object.entries(scalingResult.best_config).map(([k, v]) => (
                              <span key={k} className="mono" style={{
                                fontSize: 10,
                                padding: '3px 7px',
                                background: 'rgba(0,255,255,0.06)',
                                border: '1px solid rgba(0,255,255,0.2)',
                                borderRadius: 5,
                                color: 'var(--text-secondary)',
                              }}>
                                {k}: {v}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Compression result */}
                  {compressionResult && (
                    <div style={{
                      flex: '1 1 240px',
                      background: 'rgba(0,0,0,0.3)',
                      border: '1px solid rgba(0,255,255,0.15)',
                      borderRadius: 8,
                      padding: 14,
                    }}>
                      <div className="mono" style={{ fontSize: 9, color: 'var(--text-muted)', fontWeight: 800, letterSpacing: '1px', textTransform: 'uppercase', marginBottom: 10 }}>
                        AXIOM Compression Result
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: 5, fontSize: 12 }}>
                        <div><span style={{ color: 'var(--text-muted)' }}>Status: </span><span className="mono" style={{ color: '#00ff88' }}>{compressionResult.status}</span></div>
                        <div><span style={{ color: 'var(--text-muted)' }}>Entropy Δ: </span><span className="mono" style={{ color: 'var(--cyan)' }}>{compressionResult.entropy_delta?.toFixed(4)} bits</span></div>
                        <div><span style={{ color: 'var(--text-muted)' }}>Compression Δ: </span><span className="mono" style={{ color: 'var(--cyan)' }}>{compressionResult.compression_improvement?.toFixed(4)}</span></div>
                        <div><span style={{ color: 'var(--text-muted)' }}>Phases Done: </span><span className="mono" style={{ color: '#00ff88' }}>{compressionResult.phases_completed?.join(', ')}</span></div>
                        <div><span style={{ color: 'var(--text-muted)' }}>Phases Pending: </span><span className="mono" style={{ color: 'var(--text-secondary)' }}>{compressionResult.phases_pending?.join(', ')}</span></div>
                      </div>
                    </div>
                  )}

                </div>
              )}

            </GlassCard>
          </div>

          {/* ── Divider ── */}
          <div style={{ height: 1, background: 'linear-gradient(to right, transparent, rgba(0,255,255,0.3), transparent)', margin: '20px 0' }} />
        </div>
      )}
      </div>
      <div style={{ flex: '1 1 300px', minWidth: '300px', display: 'flex', flexDirection: 'column' }}>
        <NewsPanel 
          domain={predictions[0]?.domain || 'financial'} 
          title="Predictive Telemetry News" 
        />
      </div>
    </div>
  )
}
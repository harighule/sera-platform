import { useState, useEffect, useCallback } from 'react'
import './SecurityAssessment.css'

const API_KEY = import.meta.env.VITE_API_KEY || 'sera-demo-2026'
const BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const headers = { 'X-API-Key': API_KEY, 'Content-Type': 'application/json' }

// ─── Phase metadata ───────────────────────────────────────────────────────────
const PHASES = {
  PENDING:           { label: 'Pending',           color: '#64748b', icon: '⏳', step: 0 },
  RECON:             { label: 'Reconnaissance',     color: '#818cf8', icon: '🔍', step: 1 },
  ANALYSIS:          { label: 'Analysis',           color: '#a78bfa', icon: '🧠', step: 2 },
  VALIDATION:        { label: 'Validation',         color: '#34d399', icon: '✅', step: 3 },
  AWAITING_APPROVAL: { label: 'Awaiting Approval',  color: '#fbbf24', icon: '🔐', step: 3 },
  REPORTING:         { label: 'Reporting',          color: '#38bdf8', icon: '📋', step: 4 },
  COMPLETE:          { label: 'Complete',           color: '#10b981', icon: '🏁', step: 5 },
  ABORTED:           { label: 'Aborted',            color: '#ef4444', icon: '🛑', step: 0 },
}

const SEVERITY_META = {
  Critical: { color: '#ef4444', bg: 'rgba(239,68,68,0.15)', badge: '🔴' },
  High:     { color: '#f97316', bg: 'rgba(249,115,22,0.15)', badge: '🟠' },
  Medium:   { color: '#fbbf24', bg: 'rgba(251,191,36,0.15)', badge: '🟡' },
  Low:      { color: '#34d399', bg: 'rgba(52,211,153,0.15)', badge: '🟢' },
}

const STATUS_META = {
  confirmed_passive:              { label: 'Confirmed', color: '#10b981', icon: '✅' },
  confirmed_active:               { label: 'Approved & Confirmed', color: '#10b981', icon: '✅' },
  needs_active_exploit_to_confirm:{ label: 'Needs Approval', color: '#fbbf24', icon: '🔐' },
  rejected_false_positive:        { label: 'False Positive', color: '#64748b', icon: '❌' },
  pending:                        { label: 'Pending', color: '#818cf8', icon: '⏳' },
}

// ─── API helpers ──────────────────────────────────────────────────────────────
async function apiPost(path, body) {
  const res = await fetch(`${BASE}${path}`, { method: 'POST', headers, body: JSON.stringify(body) })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || `HTTP ${res.status}`)
  }
  return res.json()
}
async function apiGet(path) {
  const res = await fetch(`${BASE}${path}`, { headers })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

// ─── Subcomponents ─────────────────────────────────────────────────────────────
function PhaseStepper({ phase }) {
  const steps = ['RECON', 'ANALYSIS', 'VALIDATION', 'REPORTING', 'COMPLETE']
  const current = PHASES[phase]?.step ?? 0
  return (
    <div className="phase-stepper">
      {steps.map((s, i) => {
        const p = PHASES[s]
        const done = i < current
        const active = i === current - 1 || (phase === s)
        return (
          <div key={s} className={`stepper-step ${done ? 'done' : ''} ${active ? 'active' : ''}`}>
            <div className="step-dot">{done ? '✓' : p.icon}</div>
            <div className="step-label">{p.label}</div>
            {i < steps.length - 1 && <div className={`step-line ${done ? 'done' : ''}`} />}
          </div>
        )
      })}
    </div>
  )
}

function ApprovalGate({ items, engagementId, onDecision }) {
  const [decisions, setDecisions] = useState({})
  const [approver, setApprover] = useState('')
  const [submitting, setSubmitting] = useState({})

  const handleDecision = async (findingId, approved) => {
    if (!approver.trim()) return alert('Enter your operator ID before approving or denying.')
    setSubmitting(s => ({ ...s, [findingId]: true }))
    try {
      await apiPost(`/api/security/approve/${engagementId}/${findingId}`, {
        approved, approver_id: approver, notes: decisions[findingId] || ''
      })
      onDecision()
    } catch (e) {
      alert(`Approval error: ${e.message}`)
    } finally {
      setSubmitting(s => ({ ...s, [findingId]: false }))
    }
  }

  return (
    <div className="approval-gate">
      <div className="gate-header">
        <span className="gate-icon">🔐</span>
        <div>
          <h3>Human Approval Gate</h3>
          <p>The following findings require active exploitation to confirm. Review each carefully and approve or deny.</p>
        </div>
      </div>

      <div className="operator-input">
        <label>Operator ID (required for audit log)</label>
        <input value={approver} onChange={e => setApprover(e.target.value)}
          placeholder="e.g. analyst@company.com" className="input-field" />
      </div>

      {items.map(item => (
        <div key={item.finding_id} className="approval-card">
          <div className="approval-card-header">
            <span className="risk-badge" style={{ background: item.risk === 'Critical' ? '#ef444420' : '#f9731620' }}>
              {item.risk || 'High'} Risk
            </span>
            <span className="approval-target">Target: {item.target}</span>
          </div>
          <div className="approval-finding">{item.finding}</div>
          <div className="approval-meta">
            <span>Confidence: <b>{item.confidence}</b></span>
            <span>Proposed tool: <code>{item.tool}</code></span>
            <span>Proposed action: {item.proposed_action}</span>
          </div>
          <div className="approval-structured">
            <pre>{JSON.stringify({
              target: item.target,
              finding: item.finding?.substring(0, 80) + '...',
              confidence: item.confidence,
              proposed_action: item.proposed_action,
              tool: item.tool,
              risk: item.risk,
            }, null, 2)}</pre>
          </div>
          <textarea className="notes-field"
            placeholder="Operator notes for audit log..."
            value={decisions[item.finding_id] || ''}
            onChange={e => setDecisions(d => ({ ...d, [item.finding_id]: e.target.value }))} />
          <div className="approval-actions">
            <button className="btn-approve" disabled={submitting[item.finding_id]}
              onClick={() => handleDecision(item.finding_id, true)}>
              {submitting[item.finding_id] ? '...' : '✅ Approve Active Testing'}
            </button>
            <button className="btn-deny" disabled={submitting[item.finding_id]}
              onClick={() => handleDecision(item.finding_id, false)}>
              {submitting[item.finding_id] ? '...' : '❌ Deny — Mark as Not Testable'}
            </button>
          </div>
        </div>
      ))}
    </div>
  )
}

function FindingCard({ f }) {
  const [expanded, setExpanded] = useState(false)
  const sev = SEVERITY_META[f.severity] || SEVERITY_META.Low
  const status = STATUS_META[f.status] || STATUS_META.pending

  return (
    <div className="finding-card" style={{ borderLeft: `3px solid ${sev.color}` }}>
      <div className="finding-header" onClick={() => setExpanded(e => !e)}>
        <div className="finding-left">
          <span className="finding-badge" style={{ background: sev.bg, color: sev.color }}>
            {sev.badge} {f.severity || 'Unscored'}
          </span>
          <span className="finding-status" style={{ color: status.color }}>
            {status.icon} {status.label}
          </span>
          {f.cvss_score && <span className="cvss-chip">CVSS {f.cvss_score}</span>}
          {f.type === 'Zero-Input Compromise' && (
            <span className='zero-input-badge' title="Network-layer zero-input exploit">
              🌐 Zero-Input
            </span>
          )}
        </div>
        <span className="expand-toggle">{expanded ? '▲' : '▼'}</span>
      </div>

      <div className="finding-title">{f.title || f.hypothesis}</div>
      <div className="finding-hypothesis">{f.title ? f.hypothesis : ''}</div>

      {expanded && (
        <div className="finding-details">
          {f.basis && <div className="detail-block"><b>Basis:</b> {f.basis}</div>}
          {f.validation_evidence && <div className="detail-block"><b>Evidence:</b><pre>{f.validation_evidence}</pre></div>}
          {f.validation_reasoning && <div className="detail-block"><b>Reasoning:</b> {f.validation_reasoning}</div>}
          {f.description_plain && <div className="detail-block"><b>Description:</b> {f.description_plain}</div>}
          {f.business_impact && <div className="detail-block"><b>Business Impact:</b> {f.business_impact}</div>}
          {f.remediation && <div className="detail-block remediation"><b>🔧 Remediation:</b><br />{f.remediation}</div>}
          {f.cve_references?.length > 0 && (
            <div className="detail-block">
              <b>References:</b> {f.cve_references.map(r => (
                <a key={r} href={`https://nvd.nist.gov/vuln/detail/${r}`} target="_blank" rel="noreferrer"
                  className="cve-link">{r}</a>
              ))}
            </div>
          )}
          {f.owasp_category && <div className="detail-block"><b>OWASP:</b> {f.owasp_category}</div>}
          {f.cvss_vector && <div className="detail-block"><b>CVSS Vector:</b> <code>{f.cvss_vector}</code></div>}
        </div>
      )}
    </div>
  )
}

function ReportView({ report }) {
  if (!report) return null
  return (
    <div className="report-view">
      <div className="report-section exec-summary">
        <h3>📊 Executive Summary</h3>
        <p>{report.executive_summary}</p>
      </div>
      <div className="report-section">
        <h3>🎯 Scope & Methodology</h3>
        <p>{report.scope_and_methodology}</p>
      </div>
      {report.findings?.length > 0 && (
        <div className="report-section">
          <h3>🔍 Findings ({report.findings.length})</h3>
          {report.findings.map((f, i) => (
            <div key={i} className="report-finding">
              <div className="rf-header">
                <span className="rf-severity" style={{ color: SEVERITY_META[f.severity]?.color || '#fff' }}>
                  {SEVERITY_META[f.severity]?.badge} {f.severity}
                </span>
                <b>{f.title}</b>
                {f.cvss_score && <span className="cvss-chip">CVSS {f.cvss_score}</span>}
              </div>
              <p><b>Asset:</b> {f.affected_asset}</p>
              <p>{f.description}</p>
              <p><b>Evidence:</b> {f.evidence}</p>
              <p><b>Impact:</b> {f.business_impact}</p>
              <div className="rf-remediation"><b>🔧 Fix:</b> {f.remediation}</div>
              {f.cve_references?.length > 0 && (
                <p><b>Refs:</b> {f.cve_references.join(', ')} — {f.owasp_category}</p>
              )}
            </div>
          ))}
        </div>
      )}
      {report.areas_for_further_testing && (
        <div className="report-section">
          <h3>🔬 Areas for Further Authorized Testing</h3>
          <p>{report.areas_for_further_testing}</p>
        </div>
      )}
    </div>
  )
}

// ─── Main Page ─────────────────────────────────────────────────────────────────
export default function SecurityAssessment() {
  const [engagements, setEngagements] = useState([])
  const [selected, setSelected] = useState(null)
  const [report, setReport] = useState(null)
  const [loading, setLoading] = useState(false)
  const [polling, setPolling] = useState(false)
  const [form, setForm] = useState({
    target_scope: '',
    auth_reference_id: '',
    engagement_window: '',
    operator_id: '',
  })
  const [activeTab, setActiveTab] = useState('findings') // findings | report | log

  const loadEngagements = useCallback(async () => {
    try {
      const data = await apiGet('/api/security/engagements')
      setEngagements(data.engagements || [])
    } catch (e) { console.error(e) }
  }, [])

  const loadEngagement = useCallback(async (id) => {
    try {
      const data = await apiGet(`/api/security/engage/${id}`)
      setSelected(data)
    } catch (e) { console.error(e) }
  }, [])

  useEffect(() => { loadEngagements() }, [loadEngagements])

  // Poll active engagement
  useEffect(() => {
    if (!selected || ['COMPLETE', 'ABORTED'].includes(selected.phase)) return
    const interval = setInterval(() => loadEngagement(selected.id), 4000)
    setPolling(true)
    return () => { clearInterval(interval); setPolling(false) }
  }, [selected?.id, selected?.phase, loadEngagement])

  const handleCreate = async e => {
    e.preventDefault()
    if (!form.target_scope || !form.auth_reference_id || !form.engagement_window) {
      return alert('All three fields are required: target scope, authorization reference, and engagement window.')
    }
    setLoading(true)
    try {
      const result = await apiPost('/api/security/engage', form)
      await loadEngagements()
      await loadEngagement(result.engagement_id)
      setActiveTab('findings')
    } catch (e) {
      alert(`Failed to create engagement: ${e.message}`)
    } finally {
      setLoading(false)
    }
  }

  const handleGenerateReport = async () => {
    if (!selected) return
    setLoading(true)
    try {
      await apiPost(`/api/security/engage/${selected.id}/report`, {})
      setActiveTab('report')
    } catch (e) { alert(`Report error: ${e.message}`) }
    finally { setLoading(false) }
  }

  const handleLoadReport = async () => {
    if (!selected) return
    try {
      const r = await apiGet(`/api/security/engage/${selected.id}/report`)
      setReport(r)
      setActiveTab('report')
    } catch (e) { alert(`Report not ready: ${e.message}`) }
  }

  const handleAbort = async () => {
    if (!selected || !confirm('Abort this engagement?')) return
    await apiPost(`/api/security/engage/${selected.id}/abort`, {})
    await loadEngagement(selected.id)
  }

  const phase = selected ? PHASES[selected.phase] || PHASES.PENDING : null
  const awaitingApproval = selected?.approval_gate || []

  const confirmedFindings = selected?.findings?.filter(f =>
    ['confirmed_passive', 'confirmed_active'].includes(f.status)) || []
  const pendingFindings = selected?.findings?.filter(f =>
    f.status === 'needs_active_exploit_to_confirm') || []
  const rejectedFindings = selected?.findings?.filter(f =>
    f.status === 'rejected_false_positive') || []

  return (
    <div className="security-page">
      {/* ── Left Panel: Engagements ── */}
      <aside className="security-sidebar">
        <div className="sidebar-title">
          <span className="shield-icon">🛡️</span>
          <span>Assessments</span>
        </div>

        <form className="new-engagement-form" onSubmit={handleCreate}>
          <h4>New Engagement</h4>
          <div className="field-group">
            <label>Target Scope *</label>
            <input className="input-field" placeholder="e.g. 10.0.1.0/24, api.corp.com"
              value={form.target_scope} onChange={e => setForm(f => ({ ...f, target_scope: e.target.value }))} />
          </div>
          <div className="field-group">
            <label>Authorization Reference ID *</label>
            <input className="input-field" placeholder="e.g. AUTH-2026-07-001"
              value={form.auth_reference_id} onChange={e => setForm(f => ({ ...f, auth_reference_id: e.target.value }))} />
          </div>
          <div className="field-group">
            <label>Engagement Window *</label>
            <input className="input-field" placeholder="e.g. 2026-07-17 09:00–18:00 UTC"
              value={form.engagement_window} onChange={e => setForm(f => ({ ...f, engagement_window: e.target.value }))} />
          </div>
          <div className="field-group">
            <label>Operator ID</label>
            <input className="input-field" placeholder="e.g. analyst@company.com"
              value={form.operator_id} onChange={e => setForm(f => ({ ...f, operator_id: e.target.value }))} />
          </div>
          <button type="submit" className="btn-start" disabled={loading}>
            {loading ? '⏳ Starting...' : '🚀 Start Assessment'}
          </button>
        </form>

        <div className="engagement-list">
          {engagements.map(e => {
            const p = PHASES[e.phase] || PHASES.PENDING
            return (
              <div key={e.id}
                className={`engagement-item ${selected?.id === e.id ? 'active' : ''}`}
                onClick={() => { loadEngagement(e.id); setReport(null); setActiveTab('findings') }}>
                <div className="engagement-item-header">
                  <span className="phase-dot" style={{ background: p.color }}>{p.icon}</span>
                  <span className="phase-label" style={{ color: p.color }}>{p.label}</span>
                </div>
                <div className="engagement-scope">{e.target_scope?.substring(0, 40)}...</div>
                <div className="engagement-meta">
                  <span>Auth: {e.auth_reference_id}</span>
                  <span>{new Date(e.created_at).toLocaleDateString()}</span>
                </div>
              </div>
            )
          })}
          {engagements.length === 0 && (
            <div className="empty-hint">No assessments yet.<br />Create one above.</div>
          )}
        </div>
      </aside>

      {/* ── Right Panel: Engagement Detail ── */}
      <main className="security-main">
        {!selected ? (
          <div className="empty-state">
            <div className="empty-icon">🛡️</div>
            <h2>Security Assessment Console</h2>
            <p>Create a new authorized engagement or select one from the left panel.</p>
            <div className="empty-features">
              <div className="feature-item">🔍 <b>ReconAgent</b> — Passive discovery & asset mapping</div>
              <div className="feature-item">🧠 <b>AnalystAgent</b> — CVE hypothesis generation</div>
              <div className="feature-item">✅ <b>ValidatorAgent</b> — Non-destructive passive confirmation</div>
              <div className="feature-item">🔐 <b>Human Approval Gate</b> — Active exploit authorization</div>
              <div className="feature-item">📋 <b>ReportAgent</b> — Professional CVSS-scored report</div>
            </div>
          </div>
        ) : (
          <div className="engagement-detail">
            {/* Header */}
            <div className="detail-header">
              <div className="detail-title">
                <h2>{phase?.icon} {phase?.label}</h2>
                <p className="detail-scope">{selected.target_scope}</p>
              </div>
              <div className="detail-badges">
                <span className="auth-badge">Auth: {selected.auth_reference_id}</span>
                {polling && <span className="polling-badge">🔄 Live</span>}
              </div>
              <div className="detail-actions">
                {selected.phase === 'COMPLETE' && (
                  <button className="btn-report" onClick={handleLoadReport}>📥 Load Report</button>
                )}
                {['VALIDATION', 'AWAITING_APPROVAL'].includes(selected.phase) && (
                  <button className="btn-report" onClick={handleGenerateReport} disabled={loading}>
                    {loading ? '⏳...' : '📋 Generate Report'}
                  </button>
                )}
                {!['COMPLETE', 'ABORTED'].includes(selected.phase) && (
                  <button className="btn-abort" onClick={handleAbort}>🛑 Abort</button>
                )}
              </div>
            </div>

            {/* Phase stepper */}
            <PhaseStepper phase={selected.phase} />

            {/* Recon summary */}
            {selected.recon_summary && (
              <div className="recon-summary">
                <span className="summary-icon">💡</span>
                {selected.recon_summary}
              </div>
            )}

            {/* Stats row */}
            <div className="stats-row">
              <div className="stat-pill">
                <span className="stat-num">{selected.findings_count || 0}</span>
                <span className="stat-lbl">Total Hypotheses</span>
              </div>
              <div className="stat-pill confirmed">
                <span className="stat-num">{confirmedFindings.length}</span>
                <span className="stat-lbl">Confirmed</span>
              </div>
              <div className="stat-pill awaiting">
                <span className="stat-num">{pendingFindings.length}</span>
                <span className="stat-lbl">Awaiting Approval</span>
              </div>
              <div className="stat-pill rejected">
                <span className="stat-num">{rejectedFindings.length}</span>
                <span className="stat-lbl">False Positives</span>
              </div>
            </div>

            {/* Approval Gate Banner */}
            {awaitingApproval.length > 0 && (
              <div className="approval-banner" onClick={() => setActiveTab('findings')}>
                🔐 {awaitingApproval.length} finding(s) require human approval before active testing can proceed
              </div>
            )}

            {/* Tabs */}
            <div className="detail-tabs">
              {['findings', 'report', 'log'].map(t => (
                <button key={t} className={`tab-btn ${activeTab === t ? 'active' : ''}`}
                  onClick={() => setActiveTab(t)}>
                  {t === 'findings' ? '🔍 Findings' : t === 'report' ? '📋 Report' : '📜 Audit Log'}
                </button>
              ))}
            </div>

            {/* Tab: Findings */}
            {activeTab === 'findings' && (
              <div className="tab-content">
                {/* Approval gate items */}
                {awaitingApproval.length > 0 && (
                  <ApprovalGate items={awaitingApproval} engagementId={selected.id}
                    onDecision={() => loadEngagement(selected.id)} />
                )}

                {/* All findings */}
                {['RECON', 'ANALYSIS'].includes(selected.phase) && (
                  <div className="phase-running">
                    <div className="spinner" />
                    <span>Running {selected.phase === 'RECON' ? 'Reconnaissance & Analysis' : 'Analysis'}... Please wait.</span>
                  </div>
                )}
                {(selected.findings || []).length === 0 && !['RECON', 'ANALYSIS'].includes(selected.phase) && (
                  <div className="no-findings">No findings yet.</div>
                )}
                {(selected.findings || [])
                  .sort((a, b) => (a.priority || 3) - (b.priority || 3))
                  .map(f => <FindingCard key={f.id} f={f} />)
                }
              </div>
            )}

            {/* Tab: Report */}
            {activeTab === 'report' && (
              <div className="tab-content">
                {report ? (
                  <ReportView report={report} />
                ) : (
                  <div className="no-report">
                    {selected.phase === 'COMPLETE'
                      ? <><p>Report is ready.</p><button className="btn-report" onClick={handleLoadReport}>📥 Load Report</button></>
                      : <p>Report not yet generated. Generate it from the top-right button once validation is complete.</p>
                    }
                  </div>
                )}
              </div>
            )}

            {/* Tab: Audit Log */}
            {activeTab === 'log' && (
              <div className="tab-content">
                <div className="audit-log">
                  {(selected.phase_log || []).map((l, i) => (
                    <div key={i} className="log-entry">
                      <div className="log-time">{new Date(l.timestamp).toLocaleTimeString()}</div>
                      <div className="log-event-type">{l.event_type}</div>
                      <div className="log-detail">{l.from_phase} → {l.to_phase} | {l.actor}</div>
                      {l.detail && <div className="log-detail-text">{l.detail}</div>}
                    </div>
                  ))}
                  {(selected.phase_log || []).length === 0 && (
                    <div className="no-findings">No audit log entries yet.</div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  )
}

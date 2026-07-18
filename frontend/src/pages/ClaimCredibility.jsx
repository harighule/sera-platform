import { useEffect, useState, useCallback } from 'react'
import { fetchEntities, submitClaim, fetchClaim, submitChallenge, reaffirmClaim } from '../api/client'
import GlassCard from '../components/GlassCard'

const BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
const AUTH_HEADERS = { 'Content-Type': 'application/json', 'X-API-Key': 'sera-demo-2026' }

async function fetchRecentClaims() {
  try {
    const r = await fetch(`${BASE}/api/claims?limit=10`, { headers: AUTH_HEADERS })
    if (!r.ok) return null
    return await r.json()
  } catch { return null }
}

async function addEvidence(claimId, payload) {
  try {
    const r = await fetch(`${BASE}/api/claims/${claimId}/evidence`, {
      method: 'POST', headers: AUTH_HEADERS, body: JSON.stringify(payload)
    })
    if (!r.ok) return null
    return await r.json()
  } catch { return null }
}

const BAND_COLORS = {
  VERIFIED: { color: 'var(--cyan)', bg: 'rgba(0,245,212,0.07)', border: 'rgba(0,245,212,0.25)', glow: '0 0 12px rgba(0,245,212,0.25)' },
  CREDIBLE: { color: 'var(--blue)', bg: 'rgba(58,134,255,0.07)', border: 'rgba(58,134,255,0.25)', glow: '0 0 12px rgba(58,134,255,0.25)' },
  CONTESTED: { color: '#FFBE0B', bg: 'rgba(255,190,11,0.07)', border: 'rgba(255,190,11,0.25)', glow: '0 0 12px rgba(255,190,11,0.25)' },
  DISPUTED: { color: '#FF003C', bg: 'rgba(255,0,60,0.07)', border: 'rgba(255,0,60,0.25)', glow: '0 0 12px rgba(255,0,60,0.25)' },
}

function BreakdownRow({ label, value, color }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '6px 0', borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
      <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{label}</span>
      <span className="mono" style={{ fontSize: '12px', fontWeight: 'bold', color: color || 'var(--text-primary)' }}>{value}</span>
    </div>
  )
}

function StatusBadge({ band }) {
  const style = BAND_COLORS[band] || BAND_COLORS.CREDIBLE
  return (
    <span style={{
      padding: '3px 10px', borderRadius: '10px', fontSize: '10px', fontWeight: 'bold',
      letterSpacing: '0.8px', background: style.bg, color: style.color,
      border: `1px solid ${style.border}`, boxShadow: style.glow
    }}>{band}</span>
  )
}

export default function ClaimCredibility() {
  const [entities, setEntities] = useState([])
  const [recentClaims, setRecentClaims] = useState([])
  const [sessionClaimIds, setSessionClaimIds] = useState(() => {
    try { const s = localStorage.getItem('sera_session_claims'); return s ? JSON.parse(s) : [] }
    catch { return [] }
  })

  // Submit claim states
  const [claimantId, setClaimantId] = useState('')
  const [claimContent, setClaimContent] = useState('')
  const [claimStake, setClaimStake] = useState(50.0)
  const [submitMsg, setSubmitMsg] = useState('')
  const [submitting, setSubmitting] = useState(false)

  // Query/Monitor states
  const [queryClaimId, setQueryClaimId] = useState('')
  const [claimData, setClaimData] = useState(null)
  const [queryLoading, setQueryLoading] = useState(false)
  const [queryMsg, setQueryMsg] = useState('')

  // Challenge states
  const [challengerId, setChallengerId] = useState('')
  const [challengeStake, setChallengeStake] = useState(15.0)
  const [challengeText, setChallengeText] = useState('')
  const [challenging, setChallenging] = useState(false)
  const [challengeMsg, setChallengeMsg] = useState('')

  // Evidence states
  const [evidenceType, setEvidenceType] = useState('user')
  const [evidenceSource, setEvidenceSource] = useState('')
  const [evidenceContent, setEvidenceContent] = useState('')
  const [evidenceWeight, setEvidenceWeight] = useState(1.0)
  const [addingEvidence, setAddingEvidence] = useState(false)
  const [evidenceMsg, setEvidenceMsg] = useState('')

  const loadFeed = useCallback(async () => {
    const res = await fetchRecentClaims()
    if (res?.claims) setRecentClaims(res.claims)
  }, [])

  useEffect(() => {
    fetchEntities().then(res => {
      if (res?.entities) {
        setEntities(res.entities)
        if (res.entities.length > 0) {
          setClaimantId(res.entities[0].id)
          setChallengerId(res.entities[0].id)
        }
      }
    }).catch(console.error)
    loadFeed()
  }, [])

  useEffect(() => {
    if (sessionClaimIds.length > 0 && !queryClaimId) {
      setQueryClaimId(sessionClaimIds[0])
    }
  }, [sessionClaimIds])

  const handleCreateClaim = async (e) => {
    e.preventDefault()
    if (!claimContent || claimStake <= 0) { setSubmitMsg('Please provide content and a valid stake amount.'); return }
    setSubmitting(true); setSubmitMsg('')
    try {
      const res = await submitClaim({ claimant_id: claimantId, content: claimContent, stake_amount: parseFloat(claimStake) })
      if (res?.claim_id) {
        setSubmitMsg(`✓ Claim ${res.claim_id} filed! APEX: ${res.apex_verified ? '✔ Verified' : '⚬ Unverified'}`)
        setClaimContent('')
        const newIds = [res.claim_id, ...sessionClaimIds.filter(id => id !== res.claim_id)]
        setSessionClaimIds(newIds)
        localStorage.setItem('sera_session_claims', JSON.stringify(newIds))
        setQueryClaimId(res.claim_id)
        await loadClaimDetails(res.claim_id)
        loadFeed()
      } else { setSubmitMsg('Failed to submit claim.') }
    } catch (err) { console.error(err); setSubmitMsg('API request error.') }
    finally { setSubmitting(false) }
  }

  const loadClaimDetails = async (idToQuery) => {
    const targetId = idToQuery || queryClaimId
    if (!targetId) return
    setQueryLoading(true); setQueryMsg('')
    try {
      const res = await fetchClaim(targetId)
      if (res?.claim_id) {
        setClaimData(res)
      } else {
        // Silently prune stale IDs from localStorage (e.g. after DB reset)
        setSessionClaimIds(prev => {
          const updated = prev.filter(id => id !== targetId)
          localStorage.setItem('sera_session_claims', JSON.stringify(updated))
          return updated
        })
        setClaimData(null)
        setQueryClaimId('')
      }
    } catch (err) { console.error(err); setQueryMsg('Error fetching claim details.'); setClaimData(null) }
    finally { setQueryLoading(false) }
  }

  const handleReaffirm = async () => {
    if (!claimData) return
    try {
      const res = await reaffirmClaim(claimData.claim_id)
      if (res?.status === 'reaffirmed') loadClaimDetails(claimData.claim_id)
    } catch (e) { console.error(e) }
  }

  const handleChallenge = async (e) => {
    e.preventDefault()
    if (!claimData || !challengerId || challengeStake <= 0) return
    setChallenging(true); setChallengeMsg('')
    try {
      const res = await submitChallenge(claimData.claim_id, {
        challenger_id: challengerId,
        challenge_text: challengeText,
        counter_stake_amount: parseFloat(challengeStake)
      })
      if (res?.challenge_id) {
        setChallengeMsg('⚔ Challenge posted successfully!')
        setChallengeText('')
        loadClaimDetails(claimData.claim_id)
      } else { setChallengeMsg('Failed to post challenge.') }
    } catch (err) { console.error(err); setChallengeMsg('API request error.') }
    finally { setChallenging(false) }
  }

  const handleAddEvidence = async (e) => {
    e.preventDefault()
    if (!claimData || !evidenceContent) return
    setAddingEvidence(true); setEvidenceMsg('')
    try {
      const res = await addEvidence(claimData.claim_id, {
        evidence_type: evidenceType,
        source: evidenceSource,
        content: evidenceContent,
        weight: parseFloat(evidenceWeight)
      })
      if (res?.evidence_id) {
        setEvidenceMsg('✓ Evidence attached — credibility score updated.')
        setEvidenceContent(''); setEvidenceSource('')
        loadClaimDetails(claimData.claim_id)
      } else { setEvidenceMsg('Failed to attach evidence.') }
    } catch (err) { console.error(err); setEvidenceMsg('API request error.') }
    finally { setAddingEvidence(false) }
  }

  const bd = claimData?.scoring_breakdown || {}
  const bandStyle = BAND_COLORS[claimData?.status_band] || BAND_COLORS.CREDIBLE

  return (
    <div style={{ animation: 'fadeUp 0.4s ease', display: 'flex', gap: '24px', flexWrap: 'wrap', width: '100%' }}>

      {/* ─── Left column ─── */}
      <div style={{ flex: '1 1 320px', display: 'flex', flexDirection: 'column', gap: '24px' }}>

        {/* Submit claim form */}
        <GlassCard title="File Truth Attestation" subtitle="Stake-weighted claim submission with APEX causal graph verification">
          <form onSubmit={handleCreateClaim} style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
            <div>
              <label style={{ display: 'block', fontSize: '11px', color: 'var(--text-muted)', marginBottom: '6px' }}>CLAIMANT ENTITY</label>
              <select className="glass-input" value={claimantId} onChange={e => setClaimantId(e.target.value)}
                style={{ width: '100%', height: '36px', padding: '0 10px', background: 'rgba(4,7,18,0.6)' }}>
                {entities.map(e => (
                  <option key={e.id} value={e.id} style={{ background: '#0a0f24', color: 'var(--text-primary)' }}>
                    {e.name}{e.ticker ? ` [${e.ticker}]` : ` (${e.id})`}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label style={{ display: 'block', fontSize: '11px', color: 'var(--text-muted)', marginBottom: '6px' }}>CLAIM CONTENT</label>
              <textarea className="glass-input" placeholder="Enter behavioral fact or claim…"
                value={claimContent} onChange={e => setClaimContent(e.target.value)}
                style={{ width: '100%', height: '80px', padding: '10px', background: 'rgba(4,7,18,0.6)', resize: 'vertical' }} />
            </div>
            <div>
              <label style={{ display: 'block', fontSize: '11px', color: 'var(--text-muted)', marginBottom: '6px' }}>
                STAKE AMOUNT (pts): <span className="mono" style={{ color: 'var(--blue)' }}>{claimStake}</span>
              </label>
              <input type="number" className="glass-input" value={claimStake}
                onChange={e => setClaimStake(parseFloat(e.target.value))}
                style={{ width: '100%', height: '36px', padding: '0 10px', background: 'rgba(4,7,18,0.6)' }} />
            </div>
            <button type="submit" className="btn btn-primary" disabled={submitting}
              style={{ width: '100%', height: '38px', marginTop: '4px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              {submitting ? 'Submitting…' : '✓ File Claim & Run APEX Check'}
            </button>
            {submitMsg && (
              <div style={{ padding: '8px', fontSize: '11px', textAlign: 'center', borderRadius: '4px',
                background: submitMsg.includes('✓') ? 'var(--cyan-dim)' : 'var(--red-dim)',
                color: submitMsg.includes('✓') ? 'var(--cyan)' : 'var(--red)',
                border: submitMsg.includes('✓') ? '1px solid rgba(0,245,212,0.15)' : '1px solid rgba(255,0,60,0.15)' }}>
                {submitMsg}
              </div>
            )}
          </form>
        </GlassCard>

        {/* Session claims tracker */}
        <GlassCard title="Session Claim Registry" subtitle="Claims submitted in this browser session">
          {sessionClaimIds.length === 0 ? (
            <div style={{ padding: '20px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '12px' }}>No claims submitted yet.</div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              {sessionClaimIds.map(id => (
                <div key={id} onClick={() => { setQueryClaimId(id); loadClaimDetails(id) }}
                  className="table-row-hover"
                  style={{ padding: '9px 12px', borderRadius: '4px', cursor: 'pointer',
                    background: queryClaimId === id ? 'var(--blue-dim)' : 'rgba(255,255,255,0.02)',
                    border: queryClaimId === id ? '1px solid var(--blue)' : '1px solid var(--border)',
                    display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span className="mono" style={{ fontWeight: 'bold', fontSize: '12px', color: queryClaimId === id ? 'var(--blue)' : 'var(--text-primary)' }}>{id}</span>
                  <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>→ Monitor</span>
                </div>
              ))}
            </div>
          )}
        </GlassCard>

        {/* Recent claims feed */}
        <GlassCard title="Recent Claims Feed" subtitle="Latest attestations across all entities">
          {recentClaims.length === 0 ? (
            <div style={{ padding: '20px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '12px' }}>No claims on record.</div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', maxHeight: '280px', overflowY: 'auto' }}>
              {recentClaims.map(c => {
                const score = (c.credibility_score * 100).toFixed(1)
                const band = c.credibility_score >= 0.75 ? 'VERIFIED' : c.credibility_score >= 0.50 ? 'CREDIBLE' : c.credibility_score >= 0.25 ? 'CONTESTED' : 'DISPUTED'
                const bs = BAND_COLORS[band]
                return (
                  <div key={c.claim_id} onClick={() => { setQueryClaimId(c.claim_id); loadClaimDetails(c.claim_id) }}
                    className="table-row-hover"
                    style={{ padding: '9px 12px', borderRadius: '4px', cursor: 'pointer',
                      background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border)' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                      <span className="mono" style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{c.claim_id}</span>
                      <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
                        {c.apex_verified && <span style={{ fontSize: '9px', color: 'var(--cyan)', background: 'rgba(0,245,212,0.07)', padding: '1px 5px', borderRadius: '4px', border: '1px solid rgba(0,245,212,0.2)' }}>APEX ✓</span>}
                        <span style={{ fontSize: '11px', fontWeight: 'bold', fontFamily: 'monospace', color: bs.color }}>{score}%</span>
                      </div>
                    </div>
                    <div style={{ fontSize: '11px', color: 'var(--text-secondary)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      {c.claimant_name} — {c.content_preview}
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </GlassCard>
      </div>

      {/* ─── Right column ─── */}
      <div style={{ flex: '2 1 450px', display: 'flex', flexDirection: 'column', gap: '24px' }}>

        {/* Query toolbar */}
        <div className="glass-panel" style={{ padding: '16px 20px', display: 'flex', gap: '12px', alignItems: 'center' }}>
          <div style={{ flex: 1 }}>
            <input className="glass-input" placeholder="Query any Claim ID (e.g. CL-XXXXXX)…"
              value={queryClaimId} onChange={e => setQueryClaimId(e.target.value)}
              style={{ width: '100%', height: '38px' }} />
          </div>
          <button className="btn btn-primary" onClick={() => loadClaimDetails()} disabled={queryLoading || !queryClaimId}
            style={{ height: '38px', padding: '0 16px' }}>
            {queryLoading ? 'Querying…' : '🔍 Load Claim'}
          </button>
        </div>

        {queryMsg && (
          <div style={{ padding: '12px', fontSize: '13px', textAlign: 'center', borderRadius: '4px',
            background: 'var(--red-dim)', color: 'var(--red)', border: '1px solid rgba(255,0,60,0.1)' }}>
            {queryMsg}
          </div>
        )}

        {claimData && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>

            {/* ── Attestation Monitor ── */}
            <GlassCard title={`Attestation: ${claimData.claim_id}`} subtitle="ALETHEIA dynamic truth-verification telemetry">

              {/* Score + APEX badges */}
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '20px', marginBottom: '20px' }}>

                {/* Score circle */}
                <div style={{ flex: '0 0 160px', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '10px' }}>
                  <div style={{
                    width: '130px', height: '130px', borderRadius: '50%',
                    border: `3px solid ${bandStyle.border}`,
                    display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
                    boxShadow: bandStyle.glow, background: bandStyle.bg, position: 'relative'
                  }}>
                    <span style={{ fontSize: '26px', fontWeight: 'bold', color: bandStyle.color, fontFamily: 'monospace' }}>
                      {(claimData.credibility_score * 100).toFixed(1)}%
                    </span>
                    <span style={{ fontSize: '9px', color: 'var(--text-muted)', letterSpacing: '0.5px' }}>CREDIBILITY</span>
                  </div>

                  <StatusBadge band={claimData.status_band} />

                  <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', justifyContent: 'center' }}>
                    {claimData.apex_verified && (
                      <span style={{ fontSize: '10px', padding: '2px 8px', borderRadius: '8px', fontWeight: 'bold',
                        background: 'rgba(0,245,212,0.08)', color: 'var(--cyan)', border: '1px solid rgba(0,245,212,0.2)' }}>
                        ◆ APEX VERIFIED
                      </span>
                    )}
                    <span style={{ fontSize: '10px', padding: '2px 8px', borderRadius: '8px',
                      background: 'rgba(255,255,255,0.04)', color: 'var(--text-muted)', border: '1px solid var(--border)' }}>
                      {claimData.status?.toUpperCase()}
                    </span>
                  </div>

                  <button className="btn btn-primary" onClick={handleReaffirm}
                    style={{ padding: '5px 14px', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '5px',
                      background: 'rgba(0,245,212,0.08)', color: 'var(--cyan)', width: '100%', justifyContent: 'center' }}>
                    ⟳ Reaffirm Decay
                  </button>
                </div>

                {/* Scoring breakdown */}
                <div style={{ flex: '1 1 220px', display: 'flex', flexDirection: 'column', gap: '0' }}>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)', letterSpacing: '0.5px', marginBottom: '8px' }}>ALETHEIA SCORING BREAKDOWN</div>
                  <BreakdownRow label="Claimant" value={claimData.claimant_name || claimData.claimant_id} />
                  <BreakdownRow label="Stake Signal" value={bd.stake_signal?.toFixed(4)} color="var(--blue)" />
                  <BreakdownRow label="Decay Factor" value={`${(bd.decay_factor * 100).toFixed(1)}% (${bd.hours_since_reaffirm?.toFixed(1)}h)`} color={bd.decay_factor < 0.7 ? 'var(--red)' : 'var(--text-secondary)'} />
                  <BreakdownRow label="Challenge Penalty" value={`-${(bd.challenge_penalty * 100).toFixed(1)}%`} color={bd.challenge_penalty > 0 ? '#FF6B6B' : 'var(--text-muted)'} />
                  <BreakdownRow label="Evidence Boost" value={`×${bd.evidence_boost?.toFixed(3)} (Σ${bd.weighted_evidence_sum?.toFixed(2)})`} color={bd.evidence_boost > 1 ? 'var(--green)' : 'var(--text-muted)'} />
                  <BreakdownRow label="APEX Bonus" value={bd.apex_bonus === 1.15 ? '×1.150 ◆' : '×1.000'} color={bd.apex_bonus > 1 ? 'var(--cyan)' : 'var(--text-muted)'} />
                  <div style={{ marginTop: '8px', padding: '8px', background: bandStyle.bg, border: `1px solid ${bandStyle.border}`, borderRadius: '4px', display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>FINAL SCORE</span>
                    <span className="mono" style={{ fontWeight: 'bold', color: bandStyle.color }}>{claimData.credibility_score.toFixed(4)}</span>
                  </div>
                </div>
              </div>

              {/* Claim content */}
              <div style={{ background: 'rgba(255,255,255,0.02)', padding: '14px', borderRadius: '4px', border: '1px solid var(--border)', marginBottom: '10px' }}>
                <span style={{ display: 'block', fontSize: '10px', color: 'var(--text-muted)', letterSpacing: '0.5px', marginBottom: '6px' }}>ATTESTATION CONTENT</span>
                <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: '1.6', margin: 0 }}>{claimData.content}</p>
              </div>

              <div style={{ fontSize: '10px', color: 'var(--text-muted)', opacity: 0.6 }}>
                Last Reaffirmed: {new Date(claimData.last_reaffirmed_at).toLocaleString()}
              </div>
            </GlassCard>

            {/* ── Evidence + Challenges row ── */}
            <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap' }}>

              {/* Add Evidence */}
              <div style={{ flex: '1 1 240px' }}>
                <GlassCard title="Attach Evidence" subtitle="Boost credibility with supporting data">
                  <form onSubmit={handleAddEvidence} style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                    <div>
                      <label style={{ display: 'block', fontSize: '10px', color: 'var(--text-muted)', marginBottom: '5px' }}>TYPE</label>
                      <select className="glass-input" value={evidenceType} onChange={e => setEvidenceType(e.target.value)}
                        style={{ width: '100%', height: '32px', padding: '0 8px', background: 'rgba(4,7,18,0.6)', fontSize: '12px' }}>
                        <option value="financial" style={{ background: '#0a0f24' }}>Financial (+1.4×)</option>
                        <option value="graph" style={{ background: '#0a0f24' }}>Graph / APEX (+1.3×)</option>
                        <option value="document" style={{ background: '#0a0f24' }}>Document (+1.1×)</option>
                        <option value="news" style={{ background: '#0a0f24' }}>News (+1.0×)</option>
                        <option value="user" style={{ background: '#0a0f24' }}>User (+0.6×)</option>
                      </select>
                    </div>
                    <div>
                      <label style={{ display: 'block', fontSize: '10px', color: 'var(--text-muted)', marginBottom: '5px' }}>SOURCE URL / REFERENCE</label>
                      <input className="glass-input" value={evidenceSource} onChange={e => setEvidenceSource(e.target.value)}
                        placeholder="URL, ticker, or document name"
                        style={{ width: '100%', height: '32px', padding: '0 8px', background: 'rgba(4,7,18,0.6)', fontSize: '12px' }} />
                    </div>
                    <div>
                      <label style={{ display: 'block', fontSize: '10px', color: 'var(--text-muted)', marginBottom: '5px' }}>EVIDENCE CONTENT</label>
                      <textarea className="glass-input" value={evidenceContent} onChange={e => setEvidenceContent(e.target.value)}
                        placeholder="Describe the evidence supporting this claim…"
                        style={{ width: '100%', height: '60px', padding: '6px 8px', background: 'rgba(4,7,18,0.6)', resize: 'vertical', fontSize: '12px' }} />
                    </div>
                    <div>
                      <label style={{ display: 'block', fontSize: '10px', color: 'var(--text-muted)', marginBottom: '5px' }}>
                        WEIGHT: <span className="mono" style={{ color: 'var(--green)' }}>{evidenceWeight}</span>
                      </label>
                      <input type="range" min="0.1" max="1.5" step="0.1" value={evidenceWeight}
                        onChange={e => setEvidenceWeight(parseFloat(e.target.value))}
                        style={{ width: '100%', accentColor: 'var(--green)' }} />
                    </div>
                    <button type="submit" className="btn" disabled={addingEvidence}
                      style={{ width: '100%', height: '32px', background: 'rgba(57,255,20,0.07)', color: '#39FF14', border: '1px solid rgba(57,255,20,0.2)', fontSize: '12px' }}>
                      {addingEvidence ? 'Attaching…' : '+ Attach Evidence'}
                    </button>
                    {evidenceMsg && (
                      <div style={{ fontSize: '11px', textAlign: 'center', padding: '6px', borderRadius: '4px',
                        background: evidenceMsg.includes('✓') ? 'var(--cyan-dim)' : 'var(--red-dim)',
                        color: evidenceMsg.includes('✓') ? 'var(--cyan)' : 'var(--red)' }}>
                        {evidenceMsg}
                      </div>
                    )}
                  </form>
                </GlassCard>
              </div>

              {/* Post challenge */}
              <div style={{ flex: '1 1 240px' }}>
                <GlassCard title="Post Counter Challenge" subtitle="Stakes resources against this claim's credibility">
                  <form onSubmit={handleChallenge} style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                    <div>
                      <label style={{ display: 'block', fontSize: '10px', color: 'var(--text-muted)', marginBottom: '5px' }}>CHALLENGER</label>
                      <select className="glass-input" value={challengerId} onChange={e => setChallengerId(e.target.value)}
                        style={{ width: '100%', height: '32px', padding: '0 8px', background: 'rgba(4,7,18,0.6)', fontSize: '12px' }}>
                        {entities.map(e => (
                          <option key={e.id} value={e.id} style={{ background: '#0a0f24' }}>{e.name}{e.ticker ? ` [${e.ticker}]` : ''}</option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label style={{ display: 'block', fontSize: '10px', color: 'var(--text-muted)', marginBottom: '5px' }}>CHALLENGE REASON</label>
                      <textarea className="glass-input" value={challengeText} onChange={e => setChallengeText(e.target.value)}
                        placeholder="Briefly state the grounds for disputing this claim…"
                        style={{ width: '100%', height: '56px', padding: '6px 8px', background: 'rgba(4,7,18,0.6)', resize: 'vertical', fontSize: '12px' }} />
                    </div>
                    <div>
                      <label style={{ display: 'block', fontSize: '10px', color: 'var(--text-muted)', marginBottom: '5px' }}>
                        COUNTER STAKE (pts): <span className="mono" style={{ color: 'var(--magenta)' }}>{challengeStake}</span>
                      </label>
                      <input type="number" className="glass-input" value={challengeStake}
                        onChange={e => setChallengeStake(parseFloat(e.target.value))}
                        style={{ width: '100%', height: '32px', padding: '0 8px', background: 'rgba(4,7,18,0.6)', fontSize: '12px' }} />
                    </div>
                    <button type="submit" className="btn" disabled={challenging}
                      style={{ width: '100%', height: '32px', background: 'rgba(255,0,110,0.08)', color: 'var(--magenta)', border: '1px solid rgba(255,0,110,0.2)', fontSize: '12px' }}>
                      {challenging ? 'Posting…' : '⚔ Challenge This Claim'}
                    </button>
                    {challengeMsg && (
                      <div style={{ fontSize: '11px', textAlign: 'center', padding: '6px', borderRadius: '4px',
                        background: challengeMsg.includes('⚔') ? 'rgba(255,0,110,0.07)' : 'var(--red-dim)',
                        color: challengeMsg.includes('⚔') ? 'var(--magenta)' : 'var(--red)' }}>
                        {challengeMsg}
                      </div>
                    )}
                  </form>
                </GlassCard>
              </div>
            </div>

            {/* Evidence list */}
            {claimData.evidence?.length > 0 && (
              <GlassCard title="Evidence Log" subtitle={`${claimData.evidence.length} piece(s) attached`}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                  {claimData.evidence.map(ev => (
                    <div key={ev.evidence_id} style={{ padding: '10px 12px', background: 'rgba(57,255,20,0.03)', border: '1px solid rgba(57,255,20,0.1)', borderRadius: '4px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                        <span style={{ fontSize: '10px', fontWeight: 'bold', textTransform: 'uppercase', color: '#39FF14', letterSpacing: '0.5px' }}>{ev.evidence_type}</span>
                        <span className="mono" style={{ fontSize: '11px', color: 'var(--text-muted)' }}>w={ev.weight.toFixed(1)} · {ev.evidence_id}</span>
                      </div>
                      {ev.source && <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginBottom: '3px' }}>Source: {ev.source}</div>}
                      <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>{ev.content}</div>
                    </div>
                  ))}
                </div>
              </GlassCard>
            )}

            {/* Disputes list */}
            {claimData.challenges?.length > 0 && (
              <GlassCard title="Active Disputes" subtitle={`${claimData.challenges.length} challenge(s) registered`}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                  {claimData.challenges.map(c => (
                    <div key={c.challenge_id} style={{ padding: '10px 12px', background: 'rgba(255,0,110,0.03)', border: '1px solid rgba(255,0,110,0.12)', borderRadius: '4px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                        <span className="mono" style={{ fontSize: '11px', color: 'var(--magenta)', fontWeight: 'bold' }}>{c.challenge_id}</span>
                        <span className="mono" style={{ fontSize: '11px', color: 'var(--text-primary)' }}>{c.counter_stake_amount} pts</span>
                      </div>
                      <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '3px' }}>
                        Challenger: <span style={{ color: 'var(--text-secondary)' }}>{c.challenger_name || c.challenger_id}</span>
                        {' · '}<span style={{ color: c.status === 'pending' ? '#FFBE0B' : 'var(--cyan)' }}>{c.status?.toUpperCase()}</span>
                      </div>
                      {c.challenge_text && <div style={{ fontSize: '12px', color: 'var(--text-secondary)', fontStyle: 'italic' }}>"{c.challenge_text}"</div>}
                    </div>
                  ))}
                </div>
              </GlassCard>
            )}

            {/* ALETHEIA note */}
            <div className="glass-panel" style={{ padding: '12px 16px', borderLeft: '4px solid var(--amber)', background: 'rgba(255,190,11,0.03)', fontSize: '12px', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
              <strong style={{ color: 'var(--amber)', display: 'block', marginBottom: '4px' }}>ALETHEIA ENGINE NOTE</strong>
              {claimData.proof_of_concept_note}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

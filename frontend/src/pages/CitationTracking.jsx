import { useEffect, useState } from 'react'
import { fetchEntities, fetchTrackedQueries, addTrackedQuery, runCitationCheck, fetchQueryHistory, fetchEntityCitationRate } from '../api/client'
import GlassCard from '../components/GlassCard'

export default function CitationTracking() {
  const [entities, setEntities] = useState([])
  const [queries, setQueries] = useState([])
  const [loadingQueries, setLoadingQueries] = useState(false)
  
  // Submit query states
  const [queryText, setQueryText] = useState('')
  const [targetEntityName, setTargetEntityName] = useState('')
  const [submitMsg, setSubmitMsg] = useState('')
  const [submitting, setSubmitting] = useState(false)

  // Rate checker states
  const [selectedEntityForRate, setSelectedEntityForRate] = useState('')
  const [rateData, setRateData] = useState(null)
  const [rateLoading, setRateLoading] = useState(false)

  // History modal/expand states
  const [selectedQueryForHistory, setSelectedQueryForHistory] = useState(null)
  const [historyList, setHistoryList] = useState([])
  const [historyLoading, setHistoryLoading] = useState(false)
  const [checkLoadingMap, setCheckLoadingMap] = useState({})

  useEffect(() => {
    fetchEntities().then(res => {
      if (res && res.entities) {
        setEntities(res.entities)
        if (res.entities.length > 0) {
          setTargetEntityName(res.entities[0].name)
          setSelectedEntityForRate(res.entities[0].name)
        }
      }
    }).catch(e => console.error(e))

    loadQueries()
  }, [])

  const loadQueries = async () => {
    setLoadingQueries(true)
    try {
      const res = await fetchTrackedQueries()
      if (res) {
        setQueries(res)
      }
    } catch (e) {
      console.error(e)
    } finally {
      setLoadingQueries(false)
    }
  }

  useEffect(() => {
    loadCitationRate()
  }, [selectedEntityForRate])

  const loadCitationRate = async () => {
    if (!selectedEntityForRate) return
    setRateLoading(true)
    try {
      const res = await fetchEntityCitationRate(selectedEntityForRate)
      if (res) {
        setRateData(res)
      }
    } catch (e) {
      console.error(e)
    } finally {
      setRateLoading(false)
    }
  }

  const handleCreateQuery = async (e) => {
    e.preventDefault()
    if (!queryText || !targetEntityName) {
      setSubmitMsg('Please fill in both query text and target entity name.')
      return
    }
    setSubmitting(true)
    setSubmitMsg('')
    try {
      const res = await addTrackedQuery({
        query_text: queryText,
        target_entity_name: targetEntityName
      })
      if (res && (res.id || res.query_id)) {
        setSubmitMsg(`Query tracked successfully (ID: ${res.id || res.query_id})`)
        setQueryText('')
        loadQueries()
      } else {
        setSubmitMsg('Failed to add tracked query.')
      }
    } catch (err) {
      console.error(err)
      setSubmitMsg('API error.')
    } finally {
      setSubmitting(false)
    }
  }

  const handleRunCheck = async (queryId) => {
    setCheckLoadingMap(prev => ({ ...prev, [queryId]: true }))
    try {
      const res = await runCitationCheck(queryId)
      if (res) {
        loadQueries()
        loadCitationRate()
      }
    } catch (e) {
      console.error(e)
    } finally {
      setCheckLoadingMap(prev => ({ ...prev, [queryId]: false }))
    }
  }

  const handleViewHistory = async (query) => {
    setSelectedQueryForHistory(query)
    setHistoryLoading(true)
    try {
    const res = await fetchQueryHistory(query.id || query.query_id)
      if (res) {
        setHistoryList(res)
      }
    } catch (e) {
      console.error(e)
    } finally {
      setHistoryLoading(false)
    }
  }

  return (
    <div style={{ animation: 'fadeUp 0.4s ease', display: 'flex', flexDirection: 'column', gap: '24px', width: '100%' }}>
      
      {/* Alert banner */}
      <div className="glass-panel" style={{
        padding: '16px 20px',
        borderLeft: '4px solid var(--magenta)',
        background: 'rgba(255, 0, 110, 0.05)',
        fontSize: '13px',
        color: 'var(--text-primary)',
        lineHeight: '1.5',
        display: 'flex',
        flexDirection: 'column',
        gap: '6px'
      }}>
        <strong style={{ color: 'var(--magenta)', display: 'flex', alignItems: 'center', gap: '6px', fontSize: '14px' }}>
          <span>⚠</span> SIMULATED DATA SOURCE - PROOF OF CONCEPT
        </strong>
        <p style={{ color: 'var(--text-secondary)' }}>
          This feature uses a <strong>simulated citation engine</strong> logic to calculate citation rates and track voice metrics deterministically. It does NOT make actual live network calls to ChatGPT, Perplexity, or Gemini APIs.
        </p>
      </div>

      <div style={{ display: 'flex', gap: '24px', flexWrap: 'wrap' }}>
        
        {/* Left column */}
        <div style={{ flex: '1 1 350px', display: 'flex', flexDirection: 'column', gap: '24px' }}>
          
          {/* Add Tracked Query */}
          <GlassCard title="Track New SEO Query" subtitle="Define a target query to monitor in generative engine listings">
            <form onSubmit={handleCreateQuery} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div>
                <label style={{ display: 'block', fontSize: '11px', color: 'var(--text-muted)', marginBottom: '8px' }}>QUERY TEXT</label>
                <input
                  className="glass-input"
                  placeholder="e.g. Best cloud database platform for healthcare..."
                  value={queryText}
                  onChange={e => setQueryText(e.target.value)}
                  style={{ width: '100%', height: '38px', padding: '0 10px', background: 'rgba(4, 7, 18, 0.6)' }}
                />
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '11px', color: 'var(--text-muted)', marginBottom: '8px' }}>TARGET ENTITY NAME</label>
                <select
                  className="glass-input"
                  value={targetEntityName}
                  onChange={e => setTargetEntityName(e.target.value)}
                  style={{ width: '100%', height: '38px', padding: '0 10px', background: 'rgba(4, 7, 18, 0.6)' }}
                >
                  {entities.map(e => (
                    <option key={e.id} value={e.name} style={{ background: '#0a0f24', color: 'var(--text-primary)' }}>
                      {e.name}
                    </option>
                  ))}
                </select>
              </div>

              <button
                type="submit"
                className="btn btn-primary"
                disabled={submitting || !queryText}
                style={{ width: '100%', height: '38px', marginTop: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
              >
                {submitting ? 'Adding...' : '⎔ Track Query'}
              </button>

              {submitMsg && (
                <div style={{
                  padding: '10px',
                  fontSize: '11px',
                  textAlign: 'center',
                  borderRadius: '4px',
                  background: submitMsg.includes('success') ? 'var(--cyan-dim)' : 'var(--red-dim)',
                  color: submitMsg.includes('success') ? 'var(--cyan)' : 'var(--red)',
                  border: submitMsg.includes('success') ? '1px solid rgba(0, 245, 212, 0.1)' : '1px solid rgba(255, 0, 60, 0.1)'
                }}>
                  {submitMsg}
                </div>
              )}
            </form>
          </GlassCard>

          {/* Citation Rate Monitor */}
          <GlassCard title="Entity Citation Metrics" subtitle="Dynamic Share of Voice calculation across platforms">
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div>
                <label style={{ display: 'block', fontSize: '11px', color: 'var(--text-muted)', marginBottom: '8px' }}>SELECT ENTITY PROFILE</label>
                <select
                  className="glass-input"
                  value={selectedEntityForRate}
                  onChange={e => setSelectedEntityForRate(e.target.value)}
                  style={{ width: '100%', height: '38px', padding: '0 10px', background: 'rgba(4, 7, 18, 0.6)' }}
                >
                  {entities.map(e => (
                    <option key={e.id} value={e.name} style={{ background: '#0a0f24', color: 'var(--text-primary)' }}>
                      {e.name}
                    </option>
                  ))}
                </select>
              </div>

              {rateLoading ? (
                <div style={{ padding: '20px', textAlign: 'center', color: 'var(--text-muted)' }}>Calculating rate...</div>
              ) : rateData ? (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '10px 0' }}>
                  <div style={{
                    width: '130px',
                    height: '130px',
                    borderRadius: '50%',
                    border: '3px solid var(--border-bright)',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    boxShadow: '0 0 20px rgba(0, 245, 212, 0.12)',
                    marginBottom: '16px',
                    background: 'rgba(0, 245, 212, 0.02)'
                  }}>
                    <span style={{ fontSize: '28px', fontWeight: 'bold', color: 'var(--cyan)', fontFamily: 'monospace' }}>
                      {(((rateData.citation_rate ?? 0) * 100) || 0).toFixed(1)}%
                    </span>
                    <span style={{ fontSize: '9px', color: 'var(--text-muted)', letterSpacing: '0.5px', textTransform: 'uppercase' }}>CITATION RATE</span>
                  </div>

                  <div className="mono" style={{ fontSize: '12px', width: '100%', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid rgba(255,255,255,0.03)', paddingBottom: '4px' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Total Checks:</span>
                      <span style={{ color: 'var(--text-primary)', fontWeight: 'bold' }}>{rateData.total_checks}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid rgba(255,255,255,0.03)', paddingBottom: '4px' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Cited Hits:</span>
                      <span style={{ color: 'var(--blue)', fontWeight: 'bold' }}>{rateData.cited_checks}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '9px', color: 'var(--text-muted)', textAlign: 'center', marginTop: '6px' }}>
                      <span>Source: simulated_not_real_api_call</span>
                    </div>
                  </div>
                </div>
              ) : (
                <div style={{ padding: '20px', textAlign: 'center', color: 'var(--text-muted)' }}>No data loaded.</div>
              )}
            </div>
          </GlassCard>

        </div>

        {/* Right column */}
        <div style={{ flex: '2 1 500px', display: 'flex', flexDirection: 'column', gap: '24px' }}>
          
          <GlassCard title="Tracked Query Catalog" subtitle={`Active tracked query strings: ${queries.length}`}>
            {loadingQueries ? (
              <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>Loading tracked queries...</div>
            ) : queries.length === 0 ? (
              <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)', border: '1px dashed var(--border)', borderRadius: '4px' }}>
                No tracked queries configured yet. Use the form to start tracking.
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {queries.map(q => (
                  <div
                    key={q.query_id}
                    style={{
                      padding: '16px',
                      borderRadius: '6px',
                      background: 'rgba(10, 17, 36, 0.45)',
                      border: selectedQueryForHistory?.query_id === q.query_id ? '1px solid var(--blue)' : '1px solid var(--border)',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '12px'
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '12px' }}>
                      <div>
                        <span className="mono" style={{ fontSize: '10px', color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>
                          ID: {q.query_id} | Target: <strong style={{ color: 'var(--cyan)' }}>{q.target_entity_name}</strong>
                        </span>
                        <span style={{ fontSize: '14px', color: 'var(--text-primary)', fontWeight: 'bold' }}>"{q.query_text}"</span>
                      </div>
                      
                      <div style={{ display: 'flex', gap: '8px' }}>
                        <button
                          className="btn"
                          onClick={() => handleRunCheck(q.query_id)}
                          disabled={checkLoadingMap[q.query_id]}
                          style={{
                            padding: '6px 12px',
                            fontSize: '11px',
                            background: 'rgba(58, 134, 255, 0.1)',
                            color: 'var(--blue)',
                            border: '1px solid rgba(58, 134, 255, 0.2)'
                          }}
                        >
                          {checkLoadingMap[q.query_id] ? 'Checking...' : '⚡ Run Check'}
                        </button>
                        <button
                          className="btn"
                          onClick={() => handleViewHistory(q)}
                          style={{
                            padding: '6px 12px',
                            fontSize: '11px',
                            background: 'rgba(255,255,255,0.02)',
                            color: 'var(--text-secondary)',
                            border: '1px solid var(--border)'
                          }}
                        >
                          🔍 History
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </GlassCard>

          {/* History result logs */}
          {selectedQueryForHistory && (
            <GlassCard title={`Attestation Log: "${selectedQueryForHistory.query_text}"`} subtitle={`Historical citation telemetry checks`}>
              {historyLoading ? (
                <div style={{ padding: '20px', textAlign: 'center', color: 'var(--text-muted)' }}>Retrieving history...</div>
              ) : historyList.length === 0 ? (
                <div style={{ padding: '20px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '12px' }}>
                  No citation checks run yet for this query. Click "Run Check" above to trigger a check.
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                  {historyList.map(h => (
                    <div
                      key={h.result_id}
                      style={{
                        padding: '12px 14px',
                        background: 'rgba(255,255,255,0.01)',
                        border: '1px solid rgba(255,255,255,0.04)',
                        borderRadius: '4px',
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        fontSize: '12px'
                      }}
                    >
                      <div>
                        <span className="mono" style={{ textTransform: 'uppercase', color: 'var(--text-secondary)', fontWeight: 'bold' }}>{h.ai_platform}</span>
                        <span style={{ fontSize: '10px', color: 'var(--text-muted)', marginLeft: '10px' }}>{new Date(h.checked_at).toLocaleString()}</span>
                      </div>
                      
                      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                        {h.competitor_names_cited && h.competitor_names_cited.length > 0 && (
                          <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                            Competitors: {h.competitor_names_cited.join(', ')}
                          </span>
                        )}
                        <span style={{
                          padding: '3px 8px',
                          borderRadius: '10px',
                          fontSize: '10px',
                          fontWeight: 'bold',
                          textTransform: 'uppercase',
                          background: h.was_cited ? 'var(--cyan-dim)' : 'var(--red-dim)',
                          color: h.was_cited ? 'var(--cyan)' : 'var(--red)'
                        }}>
                          {h.was_cited ? 'Cited' : 'Missed'}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </GlassCard>
          )}

        </div>

      </div>

    </div>
  )
}

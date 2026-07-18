import { useEffect, useState } from 'react'
import { fetchEntities, fetchEntityConnections, createRelationship, fetchEntityMultihop } from '../api/client'
import GlassCard from '../components/GlassCard'
import ForceGraph from '../components/ForceGraph'

export default function EntityGraph() {
  const [entities, setEntities] = useState([])
  const [selectedEntityId, setSelectedEntityId] = useState('')
  const [minConfidence, setMinConfidence] = useState(0.0)
  const [connections, setConnections] = useState([])
  const [resolvedTicker, setResolvedTicker] = useState(null)
  const [loading, setLoading] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [message, setMessage] = useState('')

  // Multi-hop graph visualization state
  const [graphData, setGraphData] = useState(null)
  const [graphDepth, setGraphDepth] = useState(2)
  const loadGraph = async () => {
    if (!selectedEntityId) return
    try {
      const res = await fetchEntityMultihop(selectedEntityId, graphDepth, minConfidence)
      if (res) setGraphData(res)
    } catch (e) {
      console.error(e)
    }
  }
  useEffect(() => { loadGraph() }, [selectedEntityId, graphDepth, minConfidence])

  // Form states for creating relationship
  const [formSource, setFormSource] = useState('')
  const [formTarget, setFormTarget] = useState('')
  const [formType, setFormType] = useState('works_with')
  const [formConfidence, setFormConfidence] = useState(0.8)

  useEffect(() => {
    fetchEntities().then(res => {
      if (res && res.entities) {
        setEntities(res.entities)
        if (res.entities.length > 0) {
          setSelectedEntityId(res.entities[0].id)
          setFormSource(res.entities[0].id)
          setFormTarget(res.entities[1] ? res.entities[1].id : res.entities[0].id)
        }
      }
    }).catch(err => console.error(err))
  }, [])

  const loadConnections = async () => {
    if (!selectedEntityId) return
    setLoading(true)
    setResolvedTicker(null)
    try {
      const res = await fetchEntityConnections(selectedEntityId, minConfidence)
      if (res) {
        setConnections(res.connections || [])
        setResolvedTicker(res.ticker || null)
      }
    } catch (e) {
      console.error(e)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadConnections()
  }, [selectedEntityId, minConfidence])

  const handleAddRelationship = async (e) => {
    e.preventDefault()
    if (!formSource || !formTarget || formSource === formTarget) {
      setMessage('Source and Target entities must be distinct.')
      return
    }
    setSubmitting(true)
    setMessage('')
    try {
      const res = await createRelationship({
        source_entity_id: formSource,
        target_entity_id: formTarget,
        relationship_type: formType,
        confidence_score: parseFloat(formConfidence)
      })
      if (res && res.relationship_id) {
        setMessage('Relationship created successfully!')
        if (selectedEntityId === formSource || selectedEntityId === formTarget) {
          loadConnections()
        }
      } else {
        setMessage('Failed to create relationship.')
      }
    } catch (err) {
      console.error(err)
      setMessage('API request error.')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div style={{ animation: 'fadeUp 0.4s ease', display: 'flex', gap: '24px', flexWrap: 'wrap', width: '100%' }}>
      
      {/* 1-Hop Connections Explorer */}
      <div style={{ flex: '2 1 500px', display: 'flex', flexDirection: 'column', gap: '24px' }}>
        <GlassCard title="Multi-Hop Relationship Graph" subtitle="Force-directed visualization of the entity's N-hop neighbourhood">
          <div style={{ display: 'flex', gap: '16px', alignItems: 'center', marginBottom: '16px', flexWrap: 'wrap' }}>
            <label style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
              TRAVERSAL DEPTH: <span className="mono" style={{ color: 'var(--cyan)' }}>{graphDepth}</span>
            </label>
            <input type="range" min="1" max="4" step="1" value={graphDepth}
              onChange={e => setGraphDepth(parseInt(e.target.value))}
              style={{ flex: '1 1 160px', accentColor: 'var(--cyan)' }} />
          </div>
          <ForceGraph data={graphData} />
        </GlassCard>

        <GlassCard title="1-Hop Traversal Explorer" subtitle="Select an entity to explore direct relational connections">
          <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap', marginBottom: '24px' }}>
            <div style={{ flex: '2 1 240px' }}>
              <label style={{ display: 'block', fontSize: '11px', color: 'var(--text-muted)', marginBottom: '8px' }}>TARGET ENTITY</label>
              <select
                className="glass-input"
                value={selectedEntityId}
                onChange={e => setSelectedEntityId(e.target.value)}
                style={{ width: '100%', height: '40px', padding: '0 12px', background: 'rgba(4, 7, 18, 0.6)' }}
              >
                {entities.map(e => (
                  <option key={e.id} value={e.id} style={{ background: '#0a0f24', color: 'var(--text-primary)' }}>
                    {e.name} ({e.id})
                  </option>
                ))}
              </select>
            </div>
            
            <div style={{ flex: '1 1 180px' }}>
              <label style={{ display: 'block', fontSize: '11px', color: 'var(--text-muted)', marginBottom: '8px' }}>
                MIN CONFIDENCE: <span className="mono" style={{ color: 'var(--cyan)' }}>{minConfidence.toFixed(2)}</span>
              </label>
              <input
                type="range"
                min="0.0"
                max="1.0"
                step="0.05"
                value={minConfidence}
                onChange={e => setMinConfidence(parseFloat(e.target.value))}
                style={{ width: '100%', accentColor: 'var(--cyan)', marginTop: '8px' }}
              />
            </div>
          </div>

          {resolvedTicker && (
            <div style={{ marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '10px' }}>
              <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>QUERYING NEO4J BY TICKER:</span>
              <span className="mono" style={{ fontSize: '13px', fontWeight: 'bold', color: 'var(--cyan)', background: 'rgba(0,245,212,0.07)', padding: '2px 10px', borderRadius: '4px', border: '1px solid rgba(0,245,212,0.2)' }}>{resolvedTicker}</span>
            </div>
          )}

          {loading ? (
            <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>Querying Neo4j + SQL edges...</div>
          ) : connections.length === 0 ? (
            <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)', border: '1px dashed var(--border)', borderRadius: '4px' }}>
              No direct connections resolved for this entity matching the threshold.
              {resolvedTicker && <div style={{ marginTop: '8px', fontSize: '11px' }}>Searched Neo4j by ticker: <span className="mono" style={{ color: 'var(--cyan)' }}>{resolvedTicker}</span></div>}
            </div>
          ) : (
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px', textAlign: 'left' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border)', color: 'var(--text-muted)' }}>
                    <th style={{ padding: '12px 8px' }}>Direction</th>
                    <th style={{ padding: '12px 8px' }}>Relationship</th>
                    <th style={{ padding: '12px 8px' }}>Connected Entity</th>
                    <th style={{ padding: '12px 8px' }}>Domain</th>
                    <th style={{ padding: '12px 8px' }}>Source</th>
                    <th style={{ padding: '12px 8px', textAlign: 'right' }}>Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {connections.map(c => (
                    <tr key={c.relationship_id} style={{ borderBottom: '1px solid rgba(255,255,255,0.03)' }} className="table-row-hover">
                      <td style={{ padding: '12px 8px' }}>
                        <span style={{
                          padding: '3px 8px',
                          borderRadius: '10px',
                          fontSize: '10px',
                          fontWeight: 'bold',
                          textTransform: 'uppercase',
                          background: c.direction === 'outgoing' ? 'var(--blue-dim)' : 'var(--purple-dim)',
                          color: c.direction === 'outgoing' ? 'var(--blue)' : 'var(--purple)'
                        }}>
                          {c.direction}
                        </span>
                      </td>
                      <td style={{ padding: '12px 8px' }} className="mono">
                        {c.relationship_type}
                      </td>
                      <td style={{ padding: '12px 8px', color: 'var(--text-primary)', fontWeight: 'bold' }}>
                        {c.connected_entity.name}
                        <span style={{ fontSize: '10px', color: 'var(--text-muted)', fontWeight: 'normal', marginLeft: '6px' }}>
                          ({c.connected_entity.id})
                        </span>
                      </td>
                      <td style={{ padding: '12px 8px' }}>
                        <span style={{ fontSize: '11px', textTransform: 'capitalize' }}>{c.connected_entity.domain}</span>
                      </td>
                      <td style={{ padding: '12px 8px' }}>
                        <span style={{
                          padding: '2px 7px', borderRadius: '8px', fontSize: '10px', fontWeight: 'bold',
                          background: c.source === 'neo4j' ? 'rgba(0,245,212,0.07)' : 'rgba(58,134,255,0.07)',
                          color: c.source === 'neo4j' ? 'var(--cyan)' : 'var(--blue)',
                          border: c.source === 'neo4j' ? '1px solid rgba(0,245,212,0.2)' : '1px solid rgba(58,134,255,0.2)'
                        }}>
                          {c.source === 'neo4j' ? 'NEO4J' : 'SQL'}
                        </span>
                      </td>
                      <td style={{ padding: '12px 8px', textAlign: 'right', color: 'var(--cyan)', fontWeight: 'bold' }} className="mono">
                        {(c.confidence_score * 100).toFixed(0)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </GlassCard>
      </div>

      {/* Add New Relationship Panel */}
      <div style={{ flex: '1 1 300px', display: 'flex', flexDirection: 'column', gap: '24px' }}>
        <GlassCard title="Establish Edge Link" subtitle="Map a new semantic relationship in the local SQL database">
          <form onSubmit={handleAddRelationship} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <div>
              <label style={{ display: 'block', fontSize: '11px', color: 'var(--text-muted)', marginBottom: '8px' }}>SOURCE ENTITY</label>
              <select
                className="glass-input"
                value={formSource}
                onChange={e => setFormSource(e.target.value)}
                style={{ width: '100%', height: '38px', padding: '0 10px', background: 'rgba(4, 7, 18, 0.6)' }}
              >
                {entities.map(e => (
                  <option key={e.id} value={e.id} style={{ background: '#0a0f24', color: 'var(--text-primary)' }}>
                    {e.name} ({e.id})
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label style={{ display: 'block', fontSize: '11px', color: 'var(--text-muted)', marginBottom: '8px' }}>RELATIONSHIP TYPE</label>
              <select
                className="glass-input"
                value={formType}
                onChange={e => setFormType(e.target.value)}
                style={{ width: '100%', height: '38px', padding: '0 10px', background: 'rgba(4, 7, 18, 0.6)' }}
              >
                <option value="works_with" style={{ background: '#0a0f24' }}>works_with</option>
                <option value="co_occurs_with" style={{ background: '#0a0f24' }}>co_occurs_with</option>
                <option value="supplies_to" style={{ background: '#0a0f24' }}>supplies_to</option>
              </select>
            </div>

            <div>
              <label style={{ display: 'block', fontSize: '11px', color: 'var(--text-muted)', marginBottom: '8px' }}>TARGET ENTITY</label>
              <select
                className="glass-input"
                value={formTarget}
                onChange={e => setFormTarget(e.target.value)}
                style={{ width: '100%', height: '38px', padding: '0 10px', background: 'rgba(4, 7, 18, 0.6)' }}
              >
                {entities.map(e => (
                  <option key={e.id} value={e.id} style={{ background: '#0a0f24', color: 'var(--text-primary)' }}>
                    {e.name} ({e.id})
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label style={{ display: 'block', fontSize: '11px', color: 'var(--text-muted)', marginBottom: '8px' }}>
                CONFIDENCE SCORE: <span className="mono" style={{ color: 'var(--blue)' }}>{formConfidence}</span>
              </label>
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.05"
                value={formConfidence}
                onChange={e => setFormConfidence(parseFloat(e.target.value))}
                style={{ width: '100%', accentColor: 'var(--blue)', marginTop: '8px' }}
              />
            </div>

            <button
              type="submit"
              className="btn btn-primary"
              disabled={submitting || formSource === formTarget}
              style={{ width: '100%', height: '40px', marginTop: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
            >
              {submitting ? 'Creating Link...' : 'Create Relationship Edge'}
            </button>

            {message && (
              <div style={{
                padding: '10px',
                fontSize: '12px',
                textAlign: 'center',
                borderRadius: '4px',
                background: message.includes('success') ? 'var(--cyan-dim)' : 'var(--red-dim)',
                color: message.includes('success') ? 'var(--cyan)' : 'var(--red)',
                border: message.includes('success') ? '1px solid rgba(0, 245, 212, 0.1)' : '1px solid rgba(255, 0, 60, 0.1)'
              }}>
                {message}
              </div>
            )}
          </form>
        </GlassCard>
      </div>

    </div>
  )
}

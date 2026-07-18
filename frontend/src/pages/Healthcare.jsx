import { useEffect, useState } from 'react'
import { fetchHealthcareMetrics } from '../api/client'
import GlassCard from '../components/GlassCard'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'

export default function Healthcare() {
  const [metrics, setMetrics] = useState([])
  const [loading, setLoading] = useState(true)
  const [threshold, setThreshold] = useState(0)
  const [stats, setStats] = useState({ mean: 0, stdDev: 0, totalAdmissions: 0 })

  useEffect(() => {
    fetchHealthcareMetrics()
      .then(data => {
        if (data && data.length > 0) {
          // Calculate mean and standard deviation of admissions
          const counts = data.map(d => d.admission_count)
          const totalAdmissions = counts.reduce((a, b) => a + b, 0)
          const mean = totalAdmissions / data.length
          const variance = counts.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / data.length
          const stdDev = Math.sqrt(variance)
          const calcThreshold = mean + 2 * stdDev

          setThreshold(calcThreshold)
          setStats({ mean, stdDev, totalAdmissions })
          
          // Sort states by admission count descending for neat rendering
          const sortedData = [...data].sort((a, b) => b.admission_count - a.admission_count)
          setMetrics(sortedData)
        }
      })
      .catch(err => console.error(err))
      .finally(() => setLoading(false))
  }, [])

  const formatNumber = (num) => {
    return num ? num.toLocaleString() : '0'
  }

  const formatCurrency = (num) => {
    return num ? `$${num.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : '$0.00'
  }

  if (loading) {
    return <div className="loading-container">Synchronizing Healthcare Ingestion Stream...</div>
  }

  return (
    <div style={{ animation: 'fadeUp 0.4s ease', display: 'flex', flexDirection: 'column', gap: '24px' }}>
      
      {/* Top summary row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '20px' }}>
        <GlassCard title="Total Hospitalizations" glowType="cyan">
          <div style={{ padding: '15px 0' }}>
            <div className="mono" style={{ fontSize: '28px', color: 'var(--cyan)', fontWeight: 'bold' }}>
              {formatNumber(stats.totalAdmissions)}
            </div>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '1px', marginTop: '4px' }}>
              Aggregate NFHS-5 Records
            </div>
          </div>
        </GlassCard>

        <GlassCard title="Average Admission Rate" glowType="blue">
          <div style={{ padding: '15px 0' }}>
            <div className="mono" style={{ fontSize: '28px', color: 'var(--blue)', fontWeight: 'bold' }}>
              {formatNumber(Math.round(stats.mean))}
            </div>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '1px', marginTop: '4px' }}>
              Mean per US State
            </div>
          </div>
        </GlassCard>

        <GlassCard title="Anomalous Threshold (2 SD)" glowType="red">
          <div style={{ padding: '15px 0' }}>
            <div className="mono" style={{ fontSize: '28px', color: 'var(--red)', fontWeight: 'bold' }}>
              {formatNumber(Math.round(threshold))}
            </div>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '1px', marginTop: '4px' }}>
              Outlier Trigger Level
            </div>
          </div>
        </GlassCard>
      </div>

      {/* Chart visualization */}
      <GlassCard title="State Admission Rate Profiling" subtitle="Visualizing total discharges per region relative to the standard deviation baseline">
        <div style={{ width: '100%', height: '320px', marginTop: '20px' }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={metrics.slice(0, 20)} margin={{ top: 10, right: 10, left: 10, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
              <XAxis dataKey="region" stroke="var(--text-muted)" tick={{ fill: 'var(--text-secondary)', fontSize: 11 }} />
              <YAxis stroke="var(--text-muted)" tick={{ fill: 'var(--text-secondary)', fontSize: 11 }} />
              <Tooltip 
                contentStyle={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: '6px' }}
                labelStyle={{ color: 'var(--text-primary)', fontWeight: 'bold' }}
                itemStyle={{ color: 'var(--cyan)' }}
                formatter={(value) => [formatNumber(value), 'Hospital Admissions']}
              />
              <Bar 
                dataKey="admission_count" 
                fill="var(--cyan)" 
                radius={[4, 4, 0, 0]}
                maxBarSize={50}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </GlassCard>

      {/* Details Table */}
      <GlassCard title="NFHS-5 State Telemetry Registry" subtitle="Detailed hospitalization cost, outpatient indicators, and pharmaceutical volumes">
        <div className="table-container" style={{ marginTop: '16px' }}>
          <table className="glass-table" style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border)', color: 'var(--text-muted)', fontSize: '11px', textTransform: 'uppercase', letterSpacing: '1px' }}>
                <th style={{ padding: '12px 16px' }}>State / Region</th>
                <th style={{ padding: '12px 16px' }}>Hospital Admissions</th>
                <th style={{ padding: '12px 16px' }}>Avg Treatment Cost</th>
                <th style={{ padding: '12px 16px' }}>Drug Claim Volume</th>
                <th style={{ padding: '12px 16px', textAlign: 'center' }}>Transition Alert</th>
              </tr>
            </thead>
            <tbody>
              {metrics.map(m => {
                const isAlert = m.admission_count > threshold
                return (
                  <tr key={m.region} style={{ borderBottom: '1px solid rgba(255,255,255,0.02)' }} className="table-row-hover">
                    <td style={{ padding: '12px 16px', fontWeight: 'bold', color: 'var(--text-primary)' }}>
                      {m.region}
                    </td>
                    <td style={{ padding: '12px 16px' }} className="mono">
                      {formatNumber(m.admission_count)}
                    </td>
                    <td style={{ padding: '12px 16px' }} className="mono">
                      {formatCurrency(m.avg_total_payment)}
                    </td>
                    <td style={{ padding: '12px 16px' }} className="mono">
                      {formatNumber(m.drug_claim_count)}
                    </td>
                    <td style={{ padding: '12px 16px', textAlign: 'center' }}>
                      {isAlert ? (
                        <span 
                          style={{ 
                            background: 'var(--red-dim)', 
                            color: 'var(--red)', 
                            border: '1px solid rgba(255,0,60,0.3)',
                            padding: '4px 10px',
                            borderRadius: '4px',
                            fontSize: '10px',
                            fontWeight: 'bold',
                            letterSpacing: '1px',
                            textTransform: 'uppercase',
                            boxShadow: '0 0 10px rgba(255,0,60,0.1)'
                          }}
                        >
                          ⚠️ CRITICAL SPIKE
                        </span>
                      ) : (
                        <span 
                          style={{ 
                            background: 'var(--cyan-dim)', 
                            color: 'var(--cyan)', 
                            border: '1px solid rgba(0,245,212,0.3)',
                            padding: '4px 10px',
                            borderRadius: '4px',
                            fontSize: '10px',
                            fontWeight: 'bold',
                            letterSpacing: '1px',
                            textTransform: 'uppercase'
                          }}
                        >
                          STABLE
                        </span>
                      )}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </GlassCard>

    </div>
  )
}

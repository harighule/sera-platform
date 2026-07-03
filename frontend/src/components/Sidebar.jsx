import { useEffect, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { fetchZolaStatus } from '../api/client'

const links = [
  { path: '/', icon: '⬡', label: 'Dashboard' },
  { path: '/entities', icon: '◈', label: 'Entities' },
  { path: '/axiom', icon: '∿', label: 'AXIOM-Φ Monitor' },
  { path: '/zola', icon: '◎', label: 'ZOLA Predictions' },
  { path: '/ai', icon: '✦', label: 'AI Command' },
  { path: '/intel', icon: '⚠', label: 'Dark Intel' },
]

export default function Sidebar() {
  const location = useLocation()
  const navigate = useNavigate()
  const [zolaStatus, setZolaStatus] = useState({
    entity_mode: 'mock',
    stats: { virtual_parameters: 13000000000 },
    actual_stored_params: 0,
    wave_basis_size_kb: 0.0
  })

  useEffect(() => {
    fetchZolaStatus().then(setZolaStatus).catch(() => {})
    const i = setInterval(() => {
      fetchZolaStatus().then(setZolaStatus).catch(() => {})
    }, 8000)
    return () => clearInterval(i)
  }, [])

  const virtualParams = zolaStatus?.stats?.virtual_parameters ?? 13000000000
  const isOneQuadrillion = virtualParams >= 1e15
  const actualStoredParams = zolaStatus?.actual_stored_params ?? 0
  const waveBasisKb = zolaStatus?.wave_basis_size_kb ?? 0.0

  // Calculate percentage of 1Q parameters reached (up to 100%)
  const maxParams = 1e15
  const paramProgress = Math.min((virtualParams / maxParams) * 100, 100)

  return (
    <div className="sidebar">
      <div className="sidebar-logo">
        <h1>SERA</h1>
        <span>Intelligence Platform</span>
      </div>
      
      <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        {links.map(link => (
          <div
            key={link.path}
            className={`nav-item ${location.pathname === link.path ? 'active' : ''}`}
            onClick={() => navigate(link.path)}
          >
            <span className="nav-icon" style={{ fontSize: 18 }}>{link.icon}</span>
            <span>{link.label}</span>
          </div>
        ))}
      </div>
      
      <div className="sidebar-status">
        <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 8, letterSpacing: '1px', fontWeight: 'bold' }}>SYSTEM STATUS</div>
        <div style={{ fontSize: 12, color: 'var(--cyan)', display: 'flex', alignItems: 'center', marginBottom: 12 }}>
          <span className="status-dot" />
          <span>All Nodes Synchronized</span>
        </div>
        
        <div className="mono" style={{ fontSize: 11, color: 'var(--text-secondary)', display: 'flex', flexDirection: 'column', gap: 6, borderTop: '1px solid rgba(255,255,255,0.04)', paddingTop: 10 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span style={{ color: 'var(--text-muted)' }}>MODE:</span> 
            <span style={{ color: 'var(--cyan)', fontWeight: 'bold' }}>{(zolaStatus?.entity_mode || 'mock').toUpperCase()}</span>
          </div>
          
          <div style={{ display: 'flex', justifyContent: 'space-between', flexDirection: 'column', gap: 4 }}>
            {/* Row 1: Actual stored params */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
              <div>
                <div style={{ color: 'var(--text-muted)' }}>STORED PARAMS</div>
                <div style={{ fontSize: 9, color: 'var(--text-muted)', opacity: 0.6 }}>wave basis (compact)</div>
              </div>
              <span style={{ color: 'var(--cyan)', fontWeight: 'bold', textAlign: 'right' }}>
                {actualStoredParams.toLocaleString()}
              </span>
            </div>

            {/* Row 2: Virtual field parameters */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
              <div>
                <div style={{ color: 'var(--text-muted)' }}>TRAINED PARAMS</div>
                <div style={{ fontSize: 9, color: 'var(--text-muted)', opacity: 0.6 }}>CIFN wave basis (live)</div>
              </div>
              <span style={{ color: 'var(--blue)', fontWeight: 'bold', textAlign: 'right' }}>
                {zolaStatus?.virtual_parameters?.toLocaleString() ?? '—'}
              </span>
            </div>

            {/* Wave basis KB line */}
            <div style={{ fontSize: 9, color: 'var(--text-muted)', opacity: 0.5, textAlign: 'right' }}>
              {waveBasisKb} KB stored
            </div>

            {/* Visual Gauge Bar */}
            <div style={{ width: '100%', height: 4, background: 'rgba(255,255,255,0.06)', borderRadius: 2, overflow: 'hidden', marginTop: 2 }}>
              <div 
                style={{ 
                  height: '100%', 
                  width: `${paramProgress}%`, 
                  background: 'linear-gradient(90deg, var(--blue), var(--cyan))',
                  transition: 'width 1s cubic-bezier(0.16, 1, 0.3, 1)',
                  boxShadow: '0 0 6px var(--cyan)'
                }} 
              />
            </div>
            <div style={{ fontSize: 9, color: 'var(--text-muted)', display: 'flex', justifyContent: 'space-between', marginTop: 1 }}>
              <span>0</span>
              <span>1.0Q</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
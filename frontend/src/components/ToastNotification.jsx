import { createContext, useContext, useState, useCallback } from 'react'

const ToastContext = createContext(null)

export function useToast() {
  const context = useContext(ToastContext)
  if (!context) throw new Error('useToast must be used within ToastProvider')
  return context
}

export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([])

  const addToast = useCallback((text, type = 'info') => {
    const id = Math.random().toString(36).slice(2, 9)
    setToasts(prev => [...prev, { id, text, type }])
    setTimeout(() => {
      setToasts(prev => prev.filter(t => t.id !== id))
    }, 4500)
  }, [])

  const removeToast = useCallback((id) => {
    setToasts(prev => prev.filter(t => t.id !== id))
  }, [])

  const getBorderColor = (type) => {
    if (type === 'critical') return 'var(--red)'
    if (type === 'warning') return 'var(--amber)'
    if (type === 'success') return 'var(--cyan)'
    return 'var(--blue)'
  }

  const getIcon = (type) => {
    if (type === 'critical') return '⚡ CRITICAL'
    if (type === 'warning') return '⚠ WARNING'
    if (type === 'success') return '✓ SYSTEM'
    return 'ℹ INFO'
  }

  return (
    <ToastContext.Provider value={{ addToast }}>
      {children}
      <div 
        style={{
          position: 'fixed',
          top: 24,
          right: 24,
          zIndex: 9999,
          display: 'flex',
          flexDirection: 'column',
          gap: 12,
          maxWidth: 360,
          pointerEvents: 'none'
        }}
      >
        {toasts.map(t => (
          <div
            key={t.id}
            className="mono"
            style={{
              pointerEvents: 'auto',
              background: 'rgba(10, 15, 30, 0.85)',
              backdropFilter: 'blur(16px)',
              WebkitBackdropFilter: 'blur(16px)',
              borderLeft: `4px solid ${getBorderColor(t.type)}`,
              borderTop: '1px solid rgba(255, 255, 255, 0.05)',
              borderRight: '1px solid rgba(255, 255, 255, 0.05)',
              borderBottom: '1px solid rgba(255, 255, 255, 0.05)',
              borderRadius: 6,
              padding: '14px 18px',
              color: 'var(--text-primary)',
              fontSize: 12,
              lineHeight: '1.4',
              boxShadow: '0 8px 30px rgba(0,0,0,0.5)',
              animation: 'slideInRight 0.3s cubic-bezier(0.16, 1, 0.3, 1) both',
              display: 'flex',
              flexDirection: 'column',
              gap: 4
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontWeight: 'bold', color: getBorderColor(t.type) }}>
                {getIcon(t.type)}
              </span>
              <button 
                onClick={() => removeToast(t.id)} 
                style={{
                  background: 'transparent',
                  border: 'none',
                  color: 'var(--text-muted)',
                  cursor: 'pointer',
                  fontSize: 14,
                  padding: 0,
                  marginLeft: 16
                }}
              >
                ×
              </button>
            </div>
            <div style={{ color: 'var(--text-secondary)' }}>{t.text}</div>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  )
}

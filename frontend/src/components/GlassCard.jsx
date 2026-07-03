export default function GlassCard({ 
  children, 
  title, 
  glowType = '', // 'cyan', 'blue', 'red', or ''
  className = '', 
  style = {},
  onClick
}) {
  const glowClass = glowType ? `glowing-${glowType}` : ''
  const cursorStyle = onClick ? { cursor: 'pointer' } : {}

  return (
    <div 
      className={`card ${glowClass} ${className}`} 
      style={{ ...style, ...cursorStyle }}
      onClick={onClick}
    >
      {title && (
        <div className="card-title">
          {glowType === 'red' && <span style={{ color: 'var(--red)' }}>●</span>}
          {glowType === 'cyan' && <span style={{ color: 'var(--cyan)' }}>●</span>}
          {glowType === 'blue' && <span style={{ color: 'var(--blue)' }}>●</span>}
          {title}
        </div>
      )}
      {children}
    </div>
  )
}

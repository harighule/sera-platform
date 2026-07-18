// Lightweight dependency-free node-edge graph (radial-by-hop layout) rendered as
// SVG. Consumes the /api/graph/entity/{id}/multihop payload {nodes, edges}.
const DOMAIN_COLORS = {
    financial: '#06d6a0',
    healthcare: '#00f0ff',
    iot: '#f59e0b',
    social: '#e040fb',
    unknown: '#94a3b8',
}

export default function ForceGraph({ data, width = 560, height = 400 }) {
    if (!data || !data.nodes || data.nodes.length === 0) {
        return <div style={{ color: '#94a3b8', padding: '2rem', textAlign: 'center' }}>
            No graph data — establish edges, then render.
        </div>
    }

    const cx = width / 2
    const cy = height / 2

    // group nodes by hop distance and lay each ring out on a circle
    const byHop = {}
    data.nodes.forEach(n => { (byHop[n.hop] ??= []).push(n) })
    const pos = {}
    Object.entries(byHop).forEach(([hop, ns]) => {
        const h = Number(hop)
        const r = h === 0 ? 0 : 55 + h * 95
        ns.forEach((n, i) => {
            const a = (2 * Math.PI * i) / ns.length - Math.PI / 2
            pos[n.id] = { x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) }
        })
    })

    return (
        <svg width="100%" viewBox={`0 0 ${width} ${height}`}
            style={{ background: 'rgba(10,14,26,0.6)', borderRadius: 12, border: '1px solid rgba(255,255,255,0.08)' }}>
            {/* edges */}
            {data.edges.map((e, i) => {
                const s = pos[e.source], t = pos[e.target]
                if (!s || !t) return null
                return (
                    <g key={`e${i}`}>
                        <line x1={s.x} y1={s.y} x2={t.x} y2={t.y}
                            stroke="rgba(6,214,160,0.35)" strokeWidth={1 + 2 * (e.confidence_score || 0.5)} />
                    </g>
                )
            })}
            {/* nodes */}
            {data.nodes.map((n, i) => {
                const p = pos[n.id]
                if (!p) return null
                const color = DOMAIN_COLORS[n.domain] || DOMAIN_COLORS.unknown
                const isRoot = n.hop === 0
                return (
                    <g key={`n${i}`}>
                        <circle cx={p.x} cy={p.y} r={isRoot ? 12 : 8}
                            fill={color} fillOpacity={isRoot ? 0.95 : 0.7}
                            stroke={isRoot ? '#fff' : color} strokeWidth={isRoot ? 2 : 1} />
                        <text x={p.x} y={p.y - 14} fill="#f0f4f8" fontSize="10"
                            textAnchor="middle" style={{ fontFamily: 'monospace' }}>
                            {(n.name || n.id).slice(0, 14)}
                        </text>
                    </g>
                )
            })}
            <text x={12} y={20} fill="#94a3b8" fontSize="11" style={{ fontFamily: 'monospace' }}>
                {data.n_nodes} nodes · {data.n_edges} edges · depth {data.depth}
            </text>
        </svg>
    )
}

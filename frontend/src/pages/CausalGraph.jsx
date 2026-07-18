import React, { useState, useEffect, useRef, useCallback } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { useNavigate } from 'react-router-dom';
import './CausalGraph.css';

// â”€â”€ Sector-aware color palette â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
const SECTOR_COLORS = {
  'technology':             '#00d4ff',
  'information technology': '#00d4ff',
  'healthcare':             '#00ff88',
  'health care':            '#00ff88',
  'financials':             '#ff6b6b',
  'finance':                '#ff6b6b',
  'energy':                 '#ffa94d',
  'industrials':            '#74c0fc',
  'consumer discretionary': '#cc5de8',
  'consumer staples':       '#da77f2',
  'communication services': '#f783ac',
  'real estate':            '#e8590c',
  'materials':              '#a9e34b',
  'utilities':              '#ffe066',
};

const getNodeColor = (type, domain) => {
  if (type === 'company') {
    const sector = (domain || '').toLowerCase();
    return SECTOR_COLORS[sector] || '#00d4ff';
  }
  if (type === 'job')                                                 return '#FFDD57';
  if (type === 'news')                                                return '#FF007A';
  if (type === 'vessel' || type === 'shipping' || type === 'port')   return '#39FF14';
  return '#90A4AE';
};

const fallbackLabel = (id) => (id ? id.substring(0, 8) : '?');

const sanitizeLabel = (text) => {
  if (!text) return 'Unknown Position';
  // Remove zero-width spaces / non-printable / control characters and typical mojibake characters
  let clean = text.replace(/[\u200b-\u200d\ufeff\u200e\u200f\u00ad]/g, '');
  clean = clean.replace(/â€‹/g, ''); // Remove common mojibake zero-width space representation
  clean = clean.replace(/á€‹/g, ''); // Remove common mojibake zero-width space representation (alternate)
  clean = clean.replace(/â€[›â€˜â€™â€šâ€ž]/g, "'"); // Normalize single quotes
  clean = clean.replace(/â€¦/g, '...'); // Normalize ellipsis
  clean = clean.replace(/\s+/g, ' '); // Replace multiple spaces
  return clean.trim() || 'Unknown Position';
};

const getNodeLabel = (node) => {
  const label = node.label || (node.name && node.name !== node.id ? node.name : fallbackLabel(node.id));
  return sanitizeLabel(label);
};

const getLinkDistance = (rel) => {
  switch ((rel || '').toLowerCase()) {
    case 'mentioned_in':    return 60;
    case 'posted':          return 90;
    case 'docked_at':       return 90;
    case 'associated_with': return 150;
    default:                return 80;
  }
};

export default function CausalGraph() {
  const navigate  = useNavigate();
  const fgRef     = useRef();

  const [graphData,    setGraphData]    = useState({ nodes: [], links: [] });
  const [loading,      setLoading]      = useState(true);
  const [stabilizing,  setStabilizing]  = useState(false);
  const [searchQuery,  setSearchQuery]  = useState('');
  const [suggestions,  setSuggestions]  = useState([]);
  const [selectedNode, setSelectedNode] = useState(null);
  const [expandedNodes,setExpandedNodes]= useState(new Set());
  const [highlightIds, setHighlightIds] = useState(new Set());
  const containerRef = useRef();
  const [dims, setDims] = useState({ width: 800, height: 600 });

  // Track container dimensions for responsive graph sizing
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const obs = new ResizeObserver(entries => {
      for (const e of entries) {
        const { width, height } = e.contentRect;
        setDims({ width: Math.max(width, 200), height: Math.max(height, 200) });
      }
    });
    obs.observe(el);
    setDims({ width: el.clientWidth || 800, height: el.clientHeight || 600 });
    return () => obs.disconnect();
  }, []);

  // 1. Fetch initial company nodes
  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        const res  = await fetch('/api/semantic/companies', {
          headers: { 'X-API-Key': 'sera-demo-2026', 'Content-Type': 'application/json' }
        });
        const data = await res.json();

        const revenues = data.map(c => c.revenue || c.market_cap || 0);
        const maxRev   = Math.max(1, ...revenues);

        const initialNodes = data.map((c) => ({
          id:      c.ticker,
          label:   c.name || c.legal_name || c.ticker,
          name:    c.name || c.legal_name || c.ticker,
          type:    'company',
          domain:  c.sector || '',
          ticker:  c.ticker,
          sector:  c.sector,
          revenue: c.revenue || c.market_cap || 0,
          val:     5 + Math.min(10, ((c.revenue || 0) / maxRev) * 10),
          isCenter: true,
        }));

        setGraphData({ nodes: initialNodes, links: [] });
      } catch (err) {
        console.error('Failed to load initial graph companies:', err);
      } finally {
        setLoading(false);
      }
    };
    fetchInitialData();
  }, []);

  // 2. Apply force layout config + viewport boundary forces
  useEffect(() => {
    if (!fgRef.current || graphData.nodes.length === 0) return;
    const fg = fgRef.current;
    if (fg.d3Force) {
      fg.d3Force('charge')?.strength?.(-220);
      fg.d3Force('link')?.distance?.((link) => getLinkDistance(link.relation));
      // Pull nodes toward center to prevent overflow
      fg.d3Force('x')?.strength?.(0.06)?.x?.(dims.width / 2);
      fg.d3Force('y')?.strength?.(0.06)?.y?.(dims.height / 2);
    }
  }, [graphData.nodes.length, dims]);

  // 3. Expand adjacent nodes on company click
  const handleNodeClick = useCallback(async (node) => {
    setSelectedNode(node);
    if (fgRef.current) {
      fgRef.current.centerAt(node.x, node.y, 800);
      fgRef.current.zoom(2.5, 800);
    }
    if (node.type !== 'company' || expandedNodes.has(node.id)) return;

    try {
      const res  = await fetch(`/api/semantic/outgoing/${node.id}`, {
        headers: { 'X-API-Key': 'sera-demo-2026', 'Content-Type': 'application/json' }
      });
      const data = await res.json();
      const outgoing = data.outgoing || [];

      setGraphData((prev) => {
        const newNodes = [...prev.nodes];
        const newLinks = [...prev.links];
        const nodeMap  = new Map(newNodes.map(n => [n.id, n]));

        outgoing.forEach((rel) => {
          // Use backend-provided type; fall back to ID-prefix heuristic
          const targetType = rel.target_type && rel.target_type !== 'unknown'
            ? rel.target_type
            : rel.target.startsWith('JP-') ? 'job'
            : rel.target.match(/^(NEWS-|G-|GDELT-|MOCK-NEWS|MOCK-GDELT)/i) ? 'news'
            : 'shipping';

          if (!nodeMap.has(rel.target)) {
            // Use the real name from the backend; only fall back if missing
            const label = rel.target_name && rel.target_name !== rel.target
              ? rel.target_name
              : targetType === 'job'  ? `Job ${rel.target.replace('JP-', '').substring(0, 10)}`
              : targetType === 'news' ? `Article ${rel.target.substring(0, 8)}`
              : fallbackLabel(rel.target);

            const newNode = {
              id: rel.target, label, name: label,
              type: targetType, domain: targetType,
              val: targetType === 'news' ? 4 : targetType === 'job' ? 3.5 : 3,
              isCenter: false,
            };
            newNodes.push(newNode);
            nodeMap.set(rel.target, newNode);
          }

          const linkKey = `${node.id}-${rel.target}-${rel.relation}`;
          if (!newLinks.some(l => `${l.source?.id||l.source}-${l.target?.id||l.target}-${l.relation}` === linkKey)) {
            newLinks.push({ source: node.id, target: rel.target, relation: rel.relation, weight: rel.weight || 1 });
          }
        });

        if (newNodes.length > 100) setStabilizing(true);
        return { nodes: newNodes, links: newLinks };
      });

      setExpandedNodes(prev => { const s = new Set(prev); s.add(node.id); return s; });
    } catch (err) {
      console.error(`Failed to expand outgoing morphisms for ${node.id}:`, err);
    }
  }, [expandedNodes]);

  // 4. Search & highlight
  const handleSearchChange = (e) => {
    const q = e.target.value;
    setSearchQuery(q);
    if (!q.trim()) { setSuggestions([]); setHighlightIds(new Set()); return; }

    const ql = q.toLowerCase();
    const matched = graphData.nodes.filter(n =>
      (n.id || '').toLowerCase().includes(ql) ||
      (n.label || '').toLowerCase().includes(ql) ||
      (n.name || '').toLowerCase().includes(ql)
    );
    setHighlightIds(new Set(matched.map(n => n.id)));
    setSuggestions(matched.slice(0, 6));
  };

  const selectSuggestion = (node) => {
    setSearchQuery(getNodeLabel(node));
    setSuggestions([]);
    handleNodeClick(node);
  };

  // 5. Reset layout
  const resetLayout = () => {
    if (!fgRef.current) return;
    fgRef.current.d3ReheatSimulation?.();
    setStabilizing(true);
    setTimeout(() => setStabilizing(false), 4000);
  };

  // 6. Canvas node paint
  const nodeCanvasObject = useCallback((node, ctx, globalScale) => {
    const label    = getNodeLabel(node);
    const size     = node.val || 7;
    const color    = getNodeColor(node.type, node.domain);
    const isSelect = selectedNode?.id === node.id;
    const isHigh   = highlightIds.size > 0 && highlightIds.has(node.id);
    const isDim    = highlightIds.size > 0 && !highlightIds.has(node.id);

    if (isSelect || isHigh) {
      ctx.beginPath();
      ctx.arc(node.x, node.y, size * 1.0 + 3, 0, 2 * Math.PI);
      ctx.fillStyle = isSelect ? 'rgba(255,255,255,0.25)' : 'rgba(255,255,255,0.12)';
      ctx.fill();
    }

    ctx.shadowColor = isDim ? 'transparent' : color;
    ctx.shadowBlur  = isSelect ? 22 : (isHigh ? 14 : 7);
    ctx.fillStyle   = isDim ? 'rgba(90,90,90,0.4)' : color;
    ctx.beginPath();
    ctx.arc(node.x, node.y, size * 0.7, 0, 2 * Math.PI);
    ctx.fill();
    ctx.shadowBlur  = 0;

    const fontSize = Math.max(10 / globalScale, 4);
    ctx.font         = `${fontSize}px Inter, sans-serif`;
    ctx.textAlign    = 'center';
    ctx.textBaseline = 'top';
    const maxLabelLen  = Math.max(10, Math.floor(60 / fontSize));
    const displayLabel = label.length > maxLabelLen ? label.substring(0, maxLabelLen) + '...' : label;

    ctx.shadowColor = 'rgba(0,0,0,0.85)';
    ctx.shadowBlur  = 4;
    ctx.fillStyle   = isDim ? 'rgba(255,255,255,0.2)'
                    : (isSelect ? '#ffffff' : 'rgba(255,255,255,0.85)');
    ctx.fillText(displayLabel, node.x, node.y + size * 0.85);
    ctx.shadowBlur  = 0;
  }, [selectedNode, highlightIds]);

  const nodePointerAreaPaint = useCallback((node, color, ctx) => {
    const size = node.val || 7;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(node.x, node.y, size * 0.9, 0, 2 * Math.PI);
    ctx.fill();
  }, []);

  // 7. Canvas link paint
  const linkCanvasObject = useCallback((link, ctx) => {
    const rel   = (link.relation || '').toLowerCase();
    const start = link.source;
    const end   = link.target;
    if (!start?.x || !end?.x) return;

    ctx.beginPath();
    ctx.moveTo(start.x, start.y);
    ctx.lineTo(end.x, end.y);
    ctx.lineWidth   = Math.min((link.weight || 1) * 1.2 + 0.5, 3);
    ctx.strokeStyle = rel === 'associated_with' ? 'rgba(255,165,0,0.35)'
                    : rel === 'docked_at'       ? 'rgba(57,255,20,0.3)'
                    : 'rgba(0,200,255,0.22)';

    if (rel === 'associated_with') ctx.setLineDash([6, 4]);
    else if (rel === 'docked_at')  ctx.setLineDash([2, 3]);
    else                           ctx.setLineDash([]);

    ctx.stroke();
    ctx.setLineDash([]);
  }, []);

  useEffect(() => {
    if (!stabilizing) return;
    const t = setTimeout(() => setStabilizing(false), 4000);
    return () => clearTimeout(t);
  }, [stabilizing]);

  const activeNodeColor = selectedNode ? getNodeColor(selectedNode.type, selectedNode.domain) : 'var(--cyan)';

  return (
    <div className="causal-graph-container">

      {/* Header */}
      <div className="graph-header">
        <div>
          <h1>APEX Semantic Web Viewer</h1>
          <div className="graph-header-meta">Homotopy reasoning over live property graphs</div>
        </div>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          <span className="mono text-muted" style={{ fontSize: 12 }}>
            {graphData.nodes.length} nodes Â· {graphData.links.length} edges
          </span>
          <button className="reset-layout-btn" onClick={resetLayout}>ðŸ”„ Reset Layout</button>
        </div>
      </div>

      {/* Loading overlay */}
      {loading && (
        <div className="graph-loading-overlay">
          <div className="graph-loading-text mono">âš¡ Initializing Causal Geometry...</div>
        </div>
      )}

      {/* Stabilizing banner */}
      {stabilizing && !loading && (
        <div className="graph-stabilizing-banner mono">âš¡ Stabilizing graph layout...</div>
      )}

      {/* Left control panel */}
      <div className="graph-control-overlay">

        <div className="control-section">
          <div className="control-label">Search Entity</div>
          <div className="search-input-wrapper">
            <input
              type="text"
              placeholder="Search ticker, company, job, news..."
              className="search-input"
              value={searchQuery}
              onChange={handleSearchChange}
            />
            {suggestions.length > 0 && (
              <ul className="suggestions-list">
                {suggestions.map((s) => (
                  <li key={s.id} className="suggestion-item" onClick={() => selectSuggestion(s)}>
                    <span style={{ color: getNodeColor(s.type, s.domain), marginRight: 6 }}>â—</span>
                    {getNodeLabel(s)}
                    {s.type === 'company' && (
                      <span style={{ color: 'var(--text-muted)', marginLeft: 6, fontSize: 11 }}>[{s.id}]</span>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </div>
          {highlightIds.size > 0 && (
            <div style={{ fontSize: 11, color: 'var(--cyan)', marginTop: 4, fontFamily: 'monospace' }}>
              {highlightIds.size} node{highlightIds.size > 1 ? 's' : ''} highlighted
            </div>
          )}
        </div>

        <div className="control-section">
          <div className="control-label">Node Legend</div>
          <div className="legend-box">
            {[
              { color: '#00d4ff', label: 'Technology' },
              { color: '#00ff88', label: 'Healthcare' },
              { color: '#ff6b6b', label: 'Financials' },
              { color: '#ffa94d', label: 'Energy' },
              { color: '#cc5de8', label: 'Consumer' },
              { color: '#FFDD57', label: 'Job Posting' },
              { color: '#FF007A', label: 'News Event' },
              { color: '#39FF14', label: 'Port / Vessel' },
            ].map(({ color, label }) => (
              <div className="legend-item" key={label}>
                <div className="legend-color-dot" style={{ backgroundColor: color, boxShadow: `0 0 5px ${color}` }} />
                <span>{label}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="control-section">
          <div className="control-label">Edge Types</div>
          <div className="legend-box">
            <div className="legend-item"><div className="legend-edge-solid" /><span>mentioned_in / posted</span></div>
            <div className="legend-item"><div className="legend-edge-dashed" /><span>associated_with</span></div>
            <div className="legend-item"><div className="legend-edge-dotted" /><span>docked_at</span></div>
          </div>
        </div>

        <div className="instructions-banner mono">
          ðŸ’¡ Click company to expand. Search to highlight. Drag to explore.
        </div>
      </div>

      {/* Right node detail panel */}
      {selectedNode && (
        <div className="node-details-overlay" style={{ borderColor: `${activeNodeColor}44` }}>
          <button className="node-details-close" onClick={() => setSelectedNode(null)}>âœ•</button>

          <div className="node-type-badge" style={{ background: `${activeNodeColor}22`, color: activeNodeColor, borderColor: `${activeNodeColor}55` }}>
            {selectedNode.type.toUpperCase()}
          </div>

          <div className="node-details-title" style={{ color: activeNodeColor }}>
            {getNodeLabel(selectedNode)}
          </div>

          <div className="node-details-field">
            <span>ID</span>
            <span className="mono" style={{ fontSize: 11 }}>{selectedNode.id}</span>
          </div>

          {selectedNode.type === 'company' && <>
            {selectedNode.ticker && <div className="node-details-field"><span>Ticker</span><span>{selectedNode.ticker}</span></div>}
            {selectedNode.sector && <div className="node-details-field"><span>Sector</span><span>{selectedNode.sector}</span></div>}
            {selectedNode.revenue > 0 && (
              <div className="node-details-field">
                <span>Revenue</span><span>${(selectedNode.revenue / 1e9).toFixed(1)}B</span>
              </div>
            )}
            <div className="node-details-field">
              <span>Connections</span>
              <span>{graphData.links.filter(l => (l.source?.id||l.source) === selectedNode.id || (l.target?.id||l.target) === selectedNode.id).length}</span>
            </div>
            {!expandedNodes.has(selectedNode.id) && (
              <div style={{ fontSize: 11, color: 'var(--cyan)', marginTop: 6, fontFamily: 'monospace' }}>
                â†© Click node on graph to expand
              </div>
            )}
          </>}

          {selectedNode.type === 'job'     && <div className="node-details-field"><span>Domain</span><span>Corporate Jobs</span></div>}
          {selectedNode.type === 'news'    && <div className="node-details-field"><span>Domain</span><span>News Event</span></div>}
          {(selectedNode.type === 'vessel' || selectedNode.type === 'shipping') &&
            <div className="node-details-field"><span>Domain</span><span>Maritime Logistics</span></div>}

          <div className="node-details-divider" />

          {selectedNode.type === 'company' && (
            <button className="inspect-button" onClick={() => navigate(`/entity/${selectedNode.id}`)}>
              â†— Inspect Company Profile
            </button>
          )}
        </div>
      )}

      {/* Force-directed graph canvas */}
      <div ref={containerRef} style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
        <ForceGraph2D
          ref={fgRef}
          width={dims.width}
          height={dims.height}
          graphData={graphData}
          nodeRelVal={7}
          nodeVal={(d) => d.val}
          nodeColor={(d) => getNodeColor(d.type, d.domain)}
          linkWidth={(d) => Math.min((d.weight || 1) * 1.2 + 0.5, 3)}
          linkColor={() => 'rgba(0, 180, 220, 0.18)'}
          linkDirectionalParticles={2}
          linkDirectionalParticleSpeed={(d) => (d.weight || 1) * 0.008 + 0.002}
          linkDirectionalParticleWidth={1.5}
          linkDirectionalParticleColor={() => '#00f0ff'}
          onNodeClick={handleNodeClick}
          nodeCanvasObject={nodeCanvasObject}
          nodePointerAreaPaint={nodePointerAreaPaint}
          linkCanvasObject={linkCanvasObject}
          cooldownTicks={150}
          onEngineStop={() => setStabilizing(false)}
          d3AlphaDecay={0.02}
          d3VelocityDecay={0.3}
          minZoom={0.2}
          maxZoom={8}
        />
      </div>
    </div>
  );
}




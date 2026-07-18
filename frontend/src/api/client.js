const BASE = import.meta.env.VITE_API_BASE ?? ''
const API_KEY = import.meta.env.VITE_API_KEY ?? 'sera-demo-2026'
const AUTH_HEADERS = {
    'X-API-Key': API_KEY,
    'Content-Type': 'application/json'
}

export async function fetchStats() {
    try {
        const r = await fetch(`${BASE}/api/dashboard/stats`, { headers: AUTH_HEADERS })
        if (!r.ok) return null
        return await r.json()
    } catch (e) {
        console.error('fetchStats failed:', e)
        return null
    }
}

export async function fetchEntities({ limit, offset } = {}) {
    try {
        const url = limit !== undefined ? `${BASE}/api/entities/?limit=${limit}&offset=${offset ?? 0}` : `${BASE}/api/entities/`
        const r = await fetch(url, { headers: AUTH_HEADERS })
        if (!r.ok) return null
        return await r.json()
    } catch (e) {
        console.error('fetchEntities failed:', e)
        return null
    }
}

export async function fetchEntropy() {
    try {
        const r = await fetch(`${BASE}/api/axiom/entropy`, { headers: AUTH_HEADERS })
        if (!r.ok) return []
        const data = await r.json()
        return Array.isArray(data) ? data : []
    } catch (e) {
        console.error('fetchEntropy failed:', e)
        return []
    }
}

export async function fetchAlerts() {
    try {
        const r = await fetch(`${BASE}/api/axiom/alerts`, { headers: AUTH_HEADERS })
        if (!r.ok) return []
        const data = await r.json()
        return Array.isArray(data) ? data : []
    } catch (e) {
        console.error('fetchAlerts failed:', e)
        return []
    }
}

export async function fetchAxiomMonitor() {
    try {
        const r = await fetch(`${BASE}/api/axiom/monitor`, { headers: AUTH_HEADERS })
        if (!r.ok) return null
        return await r.json()
    } catch (e) {
        console.error('fetchAxiomMonitor failed:', e)
        return null
    }
}


export async function fetchPredictions() {
    try {
        const r = await fetch(`${BASE}/api/zola/predictions`, { headers: AUTH_HEADERS })
        if (!r.ok) return []
        const data = await r.json()
        return Array.isArray(data) ? data : []
    } catch (e) {
        console.error('fetchPredictions failed:', e)
        return []
    }
}

export async function fetchZolaDashboard() {
    try {
        const r = await fetch(`${BASE}/api/zola/dashboard`, { headers: AUTH_HEADERS })
        if (!r.ok) return null
        return await r.json()
    } catch (e) {
        console.error('fetchZolaDashboard failed:', e)
        return null
    }
}


export async function sendChat(message) {
    try {
        const r = await fetch(`${BASE}/api/chat/`, {
            method: 'POST',
            headers: AUTH_HEADERS,
            body: JSON.stringify({ message })
        })
        if (!r.ok) return { response: 'AI assistant is currently offline.' }
        return await r.json()
    } catch (e) {
        console.error('sendChat failed:', e)
        return { response: 'Connection error. Is the backend running?' }
    }
}

export async function fetchZolaStatus() {
    try {
        const r = await fetch(`${BASE}/api/zola/status`, { headers: AUTH_HEADERS })
        if (!r.ok) return null
        return await r.json()
    } catch (e) {
        console.error('fetchZolaStatus failed:', e)
        return null
    }
}

export async function triggerCyberspaceLearning() {
    try {
        const r = await fetch(`${BASE}/api/zola/learn`, {
            method: 'POST',
            headers: AUTH_HEADERS
        })
        if (!r.ok) throw new Error('API request failed')
        return await r.json()
    } catch (e) {
        console.error('triggerCyberspaceLearning failed:', e)
        throw e;
    }
}

export async function proposeSelfEvolution() {
    try {
        const r = await fetch(`${BASE}/api/zola/evolve/propose`, {
            method: 'POST',
            headers: AUTH_HEADERS
        })
        if (!r.ok) throw new Error('API request failed')
        return await r.json()
    } catch (e) {
        console.error('proposeSelfEvolution failed:', e)
        throw e;
    }
}

export async function validateSelfEvolution(patchId) {
    try {
        const r = await fetch(`${BASE}/api/zola/evolve/validate/${patchId}`, {
            method: 'POST',
            headers: AUTH_HEADERS
        })
        if (!r.ok) throw new Error('API request failed')
        return await r.json()
    } catch (e) {
        console.error('validateSelfEvolution failed:', e)
        throw e;
    }
}

export async function approveSelfEvolution(patchId) {
    try {
        const r = await fetch(`${BASE}/api/zola/evolve/approve/${patchId}`, {
            method: 'POST',
            headers: AUTH_HEADERS
        })
        if (!r.ok) throw new Error('API request failed')
        return await r.json()
    } catch (e) {
        console.error('approveSelfEvolution failed:', e)
        throw e;
    }
}

export async function fetchNews(domain = '') {
    try {
        const url = domain ? `${BASE}/api/intel/news?domain=${domain}` : `${BASE}/api/intel/news`
        const r = await fetch(url, { headers: AUTH_HEADERS })
        if (!r.ok) return []
        const data = await r.json()
        return Array.isArray(data) ? data : []
    } catch (e) {
        console.error('fetchNews failed:', e)
        return []
    }
}

export async function fetchClassified() {
    try {
        const r = await fetch(`${BASE}/api/intel/classified`, { headers: AUTH_HEADERS })
        if (!r.ok) return []
        const data = await r.json()
        return Array.isArray(data) ? data : []
    } catch (e) {
        console.error('fetchClassified failed:', e)
        return []
    }
}

export async function runKronosOptimize() {
    try {
        const r = await fetch(`${BASE}/api/zola/kronos/optimize`, {
            method: 'POST',
            headers: AUTH_HEADERS
        })
        if (!r.ok) throw new Error('API request failed')
        return await r.json()
    } catch (e) {
        console.error('runKronosOptimize failed:', e)
        throw e
    }
}

export async function fetchKronosStatus() {
    try {
        const r = await fetch(`${BASE}/api/zola/kronos/status`, { headers: AUTH_HEADERS })
        if (!r.ok) return null
        return await r.json()
    } catch (e) {
        console.error('fetchKronosStatus failed:', e)
        return null
    }
}

export async function fetchEntityArchitecture() {
    const r = await fetch(`${BASE}/api/zola/entity/architecture`, { headers: AUTH_HEADERS })
    if (!r.ok) return null
    return await r.json()
}

export async function fetchAxiomAnalysis() {
    const r = await fetch(`${BASE}/api/zola/axiom/analysis`, { headers: AUTH_HEADERS })
    if (!r.ok) return null
    return await r.json()
}

export async function triggerKronosScaling() {
    const r = await fetch(`${BASE}/api/zola/kronos/scale`, { method: 'POST', headers: AUTH_HEADERS })
    if (!r.ok) throw new Error('Scaling failed')
    return await r.json()
}

export async function getScalingStatus() {
    const r = await fetch(`${BASE}/api/zola/kronos/scale/status`, { headers: AUTH_HEADERS })
    if (!r.ok) return null
    return await r.json()
}

export async function runAxiomCompression() {
    const r = await fetch(`${BASE}/api/zola/axiom/compress`, { method: 'POST', headers: AUTH_HEADERS })
    if (!r.ok) throw new Error('Compression failed')
    return await r.json()
}

export async function getGodelAutoStatus() {
    const r = await fetch(`${BASE}/api/zola/godel/auto/status`, { headers: AUTH_HEADERS })
    if (!r.ok) return null
    return await r.json()
}

export async function fetchSynthesizedSignals(entityId) {
    try {
        const r = await fetch(`${BASE}/api/synthesize/${entityId}`, { headers: AUTH_HEADERS })
        if (!r.ok) return null
        return await r.json()
    } catch (e) {
        console.error('fetchSynthesizedSignals failed:', e)
        return null
    }
}

export async function createRelationship({ source_entity_id, target_entity_id, relationship_type, confidence_score }) {
    try {
        const r = await fetch(`${BASE}/api/graph/relationship`, {
            method: 'POST',
            headers: AUTH_HEADERS,
            body: JSON.stringify({ source_entity_id, target_entity_id, relationship_type, confidence_score })
        })
        if (!r.ok) return null
        return await r.json()
    } catch (e) {
        console.error('createRelationship failed:', e)
        return null
    }
}

export async function fetchEntityConnections(entityId, minConfidence = 0.0) {
    try {
        const r = await fetch(`${BASE}/api/graph/entity/${entityId}/connections?min_confidence=${minConfidence}`, { headers: AUTH_HEADERS })
        if (!r.ok) return null
        return await r.json()
    } catch (e) {
        console.error('fetchEntityConnections failed:', e)
        return null
    }
}

export async function submitClaim({ claimant_id, content, stake_amount }) {
    try {
        const r = await fetch(`${BASE}/api/claims`, {
            method: 'POST',
            headers: AUTH_HEADERS,
            body: JSON.stringify({ claimant_id, content, stake_amount })
        })
        if (!r.ok) return null
        return await r.json()
    } catch (e) {
        console.error('submitClaim failed:', e)
        return null
    }
}

export async function submitChallenge(claimId, { challenger_id, counter_stake_amount }) {
    try {
        const r = await fetch(`${BASE}/api/claims/${claimId}/challenge`, {
            method: 'POST',
            headers: AUTH_HEADERS,
            body: JSON.stringify({ challenger_id, counter_stake_amount })
        })
        if (!r.ok) return null
        return await r.json()
    } catch (e) {
        console.error('submitChallenge failed:', e)
        return null
    }
}

export async function reaffirmClaim(claimId) {
    try {
        const r = await fetch(`${BASE}/api/claims/${claimId}/reaffirm`, {
            method: 'POST',
            headers: AUTH_HEADERS
        })
        if (!r.ok) return null
        return await r.json()
    } catch (e) {
        console.error('reaffirmClaim failed:', e)
        return null
    }
}

export async function fetchClaim(claimId) {
    try {
        const r = await fetch(`${BASE}/api/claims/${claimId}`, { headers: AUTH_HEADERS })
        if (!r.ok) return null
        return await r.json()
    } catch (e) {
        console.error('fetchClaim failed:', e)
        return null
    }
}

export async function fetchTrackedQueries() {
    try {
        const r = await fetch(`${BASE}/api/citation/tracked`, { headers: AUTH_HEADERS })
        if (!r.ok) return null
        return await r.json()
    } catch (e) {
        console.error('fetchTrackedQueries failed:', e)
        return null
    }
}

export async function addTrackedQuery({ query_text, target_entity_name }) {
    try {
        const r = await fetch(`${BASE}/api/citation/track`, {
            method: 'POST',
            headers: AUTH_HEADERS,
            body: JSON.stringify({ query_text, target_entity_id: '', target_entity_name })
        })
        if (!r.ok) return null
        return await r.json()
    } catch (e) {
        console.error('addTrackedQuery failed:', e)
        return null
    }
}

export async function runCitationCheck(queryId) {
    try {
        const r = await fetch(`${BASE}/api/citation/run/${queryId}`, {
            method: 'POST',
            headers: AUTH_HEADERS
        })
        if (!r.ok) return null
        return await r.json()
    } catch (e) {
        console.error('runCitationCheck failed:', e)
        return null
    }
}

export async function fetchQueryHistory(queryId) {
    try {
        const r = await fetch(`${BASE}/api/citation/tracked`, { headers: AUTH_HEADERS })
        if (!r.ok) return null
        const all = await r.json()
        // Filter to just this queryId
        return Array.isArray(all) ? all.filter(q => q.id === queryId) : null
    } catch (e) {
        console.error('fetchQueryHistory failed:', e)
        return null
    }
}

export async function fetchEntityCitationRate(entityName) {
    try {
        const r = await fetch(`${BASE}/api/citation/rate?entity_name=${encodeURIComponent(entityName)}`, { headers: AUTH_HEADERS })
        if (!r.ok) return null
        return await r.json()
    } catch (e) {
        console.error('fetchEntityCitationRate failed:', e)
        return null
    }
}

export async function fetchFreshness() {
    try {
        const r = await fetch(`${BASE}/api/health/freshness`, { headers: AUTH_HEADERS })
        if (!r.ok) return null
        return await r.json()
    } catch (e) {
        console.error('fetchFreshness failed:', e)
        return null
    }
}

export async function fetchNarrativeExpansion(ticker) {
    try {
        const r = await fetch(`${BASE}/api/insights/narrative/expansion/${ticker}`, { headers: AUTH_HEADERS })
        if (!r.ok) return null
        return await r.json()
    } catch (e) {
        console.error('fetchNarrativeExpansion failed:', e)
        return null
    }
}

export async function fetchEntityFullProfile(ticker) {
    try {
        const r = await fetch(`${BASE}/api/entities/${ticker}/full`, { headers: AUTH_HEADERS })
        if (!r.ok) return null
        return await r.json()
    } catch (e) {
        console.error('fetchEntityFullProfile failed:', e)
        return null
    }
}

export async function fetchHealthcareMetrics() {
    try {
        const r = await fetch(`${BASE}/api/healthcare/metrics`, { headers: AUTH_HEADERS })
        if (!r.ok) return []
        return await r.json()
    } catch (e) {
        console.error('fetchHealthcareMetrics failed:', e)
        return []
    }
}

export async function fetchExecutiveMovements() {
    try {
        const r = await fetch(`${BASE}/api/executive/movements`, { headers: AUTH_HEADERS })
        if (!r.ok) return { movements: [], last_7_days_count: 0 }
        return await r.json()
    } catch (e) {
        console.error('fetchExecutiveMovements failed:', e)
        return { movements: [], last_7_days_count: 0 }
    }
}




export async function fetchEntityMultihop(entityId, depth = 2, minConfidence = 0.0) {
    try {
        const r = await fetch(`${BASE}/api/graph/entity/${entityId}/multihop?depth=${depth}&min_confidence=${minConfidence}`, { headers: AUTH_HEADERS })
        if (!r.ok) return null
        return await r.json()
    } catch (e) {
        console.error('fetchEntityMultihop failed:', e)
        return null
    }
}

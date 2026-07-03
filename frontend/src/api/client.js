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

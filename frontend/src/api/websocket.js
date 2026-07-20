export function createStream(onMessage) {
    try {
        // Use /api/ws prefix so Vite dev-server proxy forwards it to the
        // backend at port 8000. Plain /ws collides with Vite's own HMR socket.
        const host = window.location.host
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
        
        // Derive WS_BASE from VITE_API_BASE if available, otherwise fallback to current host
        let defaultWsBase = `${protocol}//${host}`
        const apiBase = import.meta.env.VITE_API_BASE
        if (apiBase) {
            defaultWsBase = apiBase.replace(/^http:/, 'ws:').replace(/^https:/, 'wss:')
        }
        
        const WS_BASE = import.meta.env.VITE_WS_BASE ?? defaultWsBase
        const API_KEY = import.meta.env.VITE_API_KEY ?? 'sera-demo-2026'
        
        let WS_URL;
        if (host.includes('localhost:5173') || host.includes('127.0.0.1:5173')) {
            // Development: Use the Vite proxy prefix to forward to backend HMR-safely
            WS_URL = `${WS_BASE}/api/ws/stream?api_key=${encodeURIComponent(API_KEY)}`
        } else {
            // Production: Connect directly to the backend's /ws/stream route
            WS_URL = `${WS_BASE}/ws/stream?api_key=${encodeURIComponent(API_KEY)}`
        }
        
        const ws = new WebSocket(WS_URL)
        ws.onmessage = (e) => {
            try {
                const data = JSON.parse(e.data)
                onMessage(data)
            } catch {}
        }
        ws.onerror = (err) => console.warn('[SERA WS] connection error — live stream inactive', err)
        ws.onclose = (e) => {
            // Don't spam reconnect on auth rejection (1008)
            if (e.code === 1008) return
            console.info(`[SERA WS] closed (code ${e.code}) — stream will reconnect on next page load`)
        }
        return ws
    } catch (e) {
        console.error('WebSocket connection failed:', e)
        return null;
    }
}
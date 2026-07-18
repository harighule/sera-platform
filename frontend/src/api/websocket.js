export function createStream(onMessage) {
    try {
        // Use /api/ws prefix so Vite dev-server proxy forwards it to the
        // backend at port 8000. Plain /ws collides with Vite's own HMR socket.
        const host = window.location.host
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
        const WS_BASE = import.meta.env.VITE_WS_BASE ?? `${protocol}//${host}`
        const API_KEY = import.meta.env.VITE_API_KEY ?? 'sera-demo-2026'
        const WS_URL = `${WS_BASE}/api/ws/stream?api_key=${encodeURIComponent(API_KEY)}`
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
export function createStream(onMessage) {
    try {
        const WS_BASE = import.meta.env.VITE_WS_BASE ?? `ws://${window.location.host}`
        const WS_URL = `${WS_BASE}/ws/stream`
        const ws = new WebSocket(WS_URL)
        ws.onmessage = (e) => {
            try {
                const data = JSON.parse(e.data)
                onMessage(data)
            } catch {}
        }
        ws.onerror = () => console.warn('WebSocket error')
        return ws
    } catch (e) {
        console.error('WebSocket connection failed:', e)
        return null;
    }
}
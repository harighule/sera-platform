import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import Header from './components/Header'
import Dashboard from './pages/Dashboard'
import Entities from './pages/Entities'
import AxiomMonitor from './pages/AxiomMonitor'
import ZolaPredictions from './pages/ZolaPredictions'
import AIAssistant from './pages/AIAssistant'
import DarkIntel from './pages/DarkIntel'
import ParticleBackground from './components/ParticleBackground'
import { ToastProvider } from './components/ToastNotification'

const pages = {
  '/': { title: 'SERA Dashboard', subtitle: 'Real-time multi-protocol intelligence overview' },
  '/entities': { title: 'Entity Registry', subtitle: 'Resolved profiles & state stability registers' },
  '/axiom': { title: 'AXIOM-Φ Monitor', subtitle: 'Shannon entropy & pre-transition detection alerts' },
  '/zola': { title: 'ZOLA Causal Engine', subtitle: 'Behavioral predictions & KRONOS self-evolution' },
  '/ai': { title: 'AI Command Console', subtitle: 'Natural language interface to platform subsystems' },
  '/intel': { title: 'Dark Intel briefings', subtitle: 'Classified behavioral intelligence briefings (Clearance Required)' }
}

function Layout({ path, children }) {
  const meta = pages[path] || pages['/']
  return (
    <div className='app-layout'>
      <Sidebar />
      <Header title={meta.title} subtitle={meta.subtitle} />
      <main className='main-content'>{children}</main>
    </div>
  )
}

export default function App() {
  return (
    <ToastProvider>
      <BrowserRouter>
        <ParticleBackground />
        <Routes>
          <Route path="/" element={<Layout path="/"><Dashboard /></Layout>} />
          <Route path="/entities" element={<Layout path="/entities"><Entities /></Layout>} />
          <Route path="/axiom" element={<Layout path="/axiom"><AxiomMonitor /></Layout>} />
          <Route path="/zola" element={<Layout path="/zola"><ZolaPredictions /></Layout>} />
          <Route path="/ai" element={<Layout path="/ai"><AIAssistant /></Layout>} />
          <Route path="/intel" element={<Layout path="/intel"><DarkIntel /></Layout>} />
        </Routes>
      </BrowserRouter>
    </ToastProvider>
  )
}
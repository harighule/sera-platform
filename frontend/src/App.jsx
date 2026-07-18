import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import Header from './components/Header'
import Dashboard from './pages/Dashboard'
import Entities from './pages/Entities'
import AxiomMonitor from './pages/AxiomMonitor'
import ZolaPredictions from './pages/ZolaPredictions'
import AIAssistant from './pages/AIAssistant'
import DarkIntel from './pages/DarkIntel'
import SignalSynthesis from './pages/SignalSynthesis'
import EntityGraph from './pages/EntityGraph'
import ClaimCredibility from './pages/ClaimCredibility'
import CitationTracking from './pages/CitationTracking'
import EntityDetail from './pages/EntityDetail'
import CausalGraph from './pages/CausalGraph'
import Healthcare from './pages/Healthcare'
import Executive from './pages/Executive'
import SecurityAssessment from './pages/SecurityAssessment'
import ParticleBackground from './components/ParticleBackground'
import { ToastProvider } from './components/ToastNotification'

const pages = {
  '/': { title: 'SERA Dashboard', subtitle: 'Real-time multi-protocol intelligence overview' },
  '/entities': { title: 'Entity Registry', subtitle: 'Resolved profiles & state stability registers' },
  '/entity/:ticker': { title: 'Corporate Intelligence Briefing', subtitle: 'Dynamic 360° entity profiling & predictive insights' },
  '/synthesize': { title: 'Signal Synthesis Console', subtitle: 'Synthesized multi-source causal intelligence fusion' },
  '/graph': { title: 'Entity Knowledge Graph', subtitle: 'Semantic relationship registry & 1-hop traversals' },
  '/claims': { title: 'Claim Credibility (ALETHEIA)', subtitle: 'Stake-weighted adversarial truth verification' },
  '/geo': { title: 'Citation Tracking (GEO)', subtitle: 'Generative Engine Optimization citation Share of Voice' },
  '/axiom': { title: 'AXIOM-Φ Monitor', subtitle: 'Shannon entropy & pre-transition detection alerts' },
  '/zola': { title: 'ZOLA Causal Engine', subtitle: 'Behavioral predictions & KRONOS self-evolution' },
  '/ai': { title: 'AI Command Console', subtitle: 'Natural language interface to platform subsystems' },
  '/intel': { title: 'Dark Intel briefings', subtitle: 'Classified behavioral intelligence briefings (Clearance Required)' },
  '/causal-graph': { title: 'APEX Causal Geometry', subtitle: 'Interactive force-directed property graph visualizations' },
  '/healthcare': { title: 'Healthcare CMS Dashboard', subtitle: 'Hospital admissions, Medicare spending, and pharmaceutical metrics by state' },
  '/executive': { title: 'Executive Intelligence Briefing', subtitle: 'Public corporate leadership transitions and alignments from LinkedIn' },
  '/security': { title: 'Security Assessment Console', subtitle: 'Multi-agent authorized pentest pipeline — Recon → Analysis → Validation → Human Approval → Report' }
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
          <Route path="/entity/:ticker" element={<Layout path="/entity/:ticker"><EntityDetail /></Layout>} />
          <Route path="/synthesize" element={<Layout path="/synthesize"><SignalSynthesis /></Layout>} />
          <Route path="/graph" element={<Layout path="/graph"><EntityGraph /></Layout>} />
          <Route path="/claims" element={<Layout path="/claims"><ClaimCredibility /></Layout>} />
          <Route path="/geo" element={<Layout path="/geo"><CitationTracking /></Layout>} />
          <Route path="/axiom" element={<Layout path="/axiom"><AxiomMonitor /></Layout>} />
          <Route path="/zola" element={<Layout path="/zola"><ZolaPredictions /></Layout>} />
          <Route path="/ai" element={<Layout path="/ai"><AIAssistant /></Layout>} />
          <Route path="/intel" element={<Layout path="/intel"><DarkIntel /></Layout>} />
          <Route path="/causal-graph" element={<Layout path="/causal-graph"><CausalGraph /></Layout>} />
          <Route path="/healthcare" element={<Layout path="/healthcare"><Healthcare /></Layout>} />
          <Route path="/executive" element={<Layout path="/executive"><Executive /></Layout>} />
          <Route path="/security" element={<Layout path="/security"><SecurityAssessment /></Layout>} />
        </Routes>
      </BrowserRouter>
    </ToastProvider>
  )
}
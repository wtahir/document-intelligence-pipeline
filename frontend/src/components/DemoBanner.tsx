import { Zap } from 'lucide-react'
import { useConfig } from '../context/ConfigContext'

export default function DemoBanner() {
  const { demo_mode } = useConfig()
  if (!demo_mode) return null

  return (
    <div className="bg-brand-500/8 border-b border-brand-500/20 px-6 py-2.5 flex items-center gap-3 text-sm">
      <Zap size={14} className="text-brand-400 shrink-0" />
      <span className="text-surface-300">
        <strong className="text-brand-400">Agentic RAG Demo</strong>
        <span className="hidden sm:inline"> — Production pipeline with Query Intelligence, Knowledge Graph, Context Engineering &amp; Self-Critique. Pre-computed results from 90 insurance claim documents.</span>
        <span className="sm:hidden"> — Agentic pipeline with KG + self-correction.</span>
      </span>
    </div>
  )
}

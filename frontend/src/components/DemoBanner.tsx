import { Info } from 'lucide-react'
import { useConfig } from '../context/ConfigContext'

export default function DemoBanner() {
  const { demo_mode } = useConfig()
  if (!demo_mode) return null

  return (
    <div className="bg-amber-500/10 border-b border-amber-500/20 px-6 py-2 flex items-center gap-2 text-sm text-amber-400">
      <Info size={14} className="shrink-0" />
      <span>
        <strong>Demo mode</strong> — pre-baked water/storm/glass claim data.
        Pipeline execution and live RAG queries are disabled.
      </span>
    </div>
  )
}

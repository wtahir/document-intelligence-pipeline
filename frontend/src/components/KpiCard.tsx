interface KpiCardProps {
  label: string
  value: string | number | null | undefined
  sub?: string
  accent?: boolean
}

export default function KpiCard({ label, value, sub, accent }: KpiCardProps) {
  const display = value === null || value === undefined ? '—' : value
  return (
    <div className={`card p-5 ${accent ? 'border-brand-500/40 bg-brand-500/5' : ''}`}>
      <div className="text-xs font-semibold text-surface-400 uppercase tracking-wider mb-2">{label}</div>
      <div className={`text-3xl font-bold ${accent ? 'text-brand-400' : 'text-surface-50'}`}>
        {display}
      </div>
      {sub && <div className="mt-1 text-xs text-surface-500">{sub}</div>}
    </div>
  )
}

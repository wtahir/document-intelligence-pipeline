type StatusType = 'success' | 'failed' | 'skipped' | 'complete' | 'partial' | 'not_run' | 'running' | string

const CONFIG: Record<string, { bg: string; text: string; label: string }> = {
  success:   { bg: 'bg-emerald-500/15', text: 'text-emerald-400', label: 'Success' },
  complete:  { bg: 'bg-emerald-500/15', text: 'text-emerald-400', label: 'Complete' },
  failed:    { bg: 'bg-red-500/15',     text: 'text-red-400',     label: 'Failed' },
  partial:   { bg: 'bg-amber-500/15',   text: 'text-amber-400',   label: 'Partial' },
  skipped:   { bg: 'bg-amber-500/15',   text: 'text-amber-400',   label: 'Skipped' },
  not_run:   { bg: 'bg-surface-700',    text: 'text-surface-400', label: 'Not Run' },
  running:   { bg: 'bg-blue-500/15',    text: 'text-blue-400',    label: 'Running' },
}

export default function StatusBadge({ status }: { status: StatusType }) {
  const cfg = CONFIG[status] ?? CONFIG.not_run
  return (
    <span className={`inline-block px-2 py-0.5 rounded-full text-xs font-semibold ${cfg.bg} ${cfg.text}`}>
      {cfg.label}
    </span>
  )
}

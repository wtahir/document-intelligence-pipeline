interface PageHeaderProps {
  title: string
  subtitle?: string
  badge?: string
  actions?: React.ReactNode
}

export default function PageHeader({ title, subtitle, badge, actions }: PageHeaderProps) {
  return (
    <div className="px-8 pt-8 pb-6 border-b border-surface-700/50">
      <div className="flex items-start justify-between gap-4">
        <div>
          {badge && (
            <span className="inline-block mb-2 px-2.5 py-0.5 rounded-full text-xs font-semibold
                             bg-brand-500/15 text-brand-400 border border-brand-500/20">
              {badge}
            </span>
          )}
          <h1 className="text-2xl font-bold text-surface-50">{title}</h1>
          {subtitle && (
            <p className="mt-1 text-sm text-surface-400 max-w-2xl">{subtitle}</p>
          )}
        </div>
        {actions && <div className="flex items-center gap-2 shrink-0">{actions}</div>}
      </div>
    </div>
  )
}

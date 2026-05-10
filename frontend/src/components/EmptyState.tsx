import { AlertCircle } from 'lucide-react'

interface EmptyStateProps {
  title: string
  description?: string
}

export default function EmptyState({ title, description }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center py-24 text-center">
      <AlertCircle size={40} className="text-surface-600 mb-4" />
      <div className="text-lg font-semibold text-surface-400">{title}</div>
      {description && <div className="mt-1 text-sm text-surface-600 max-w-sm">{description}</div>}
    </div>
  )
}

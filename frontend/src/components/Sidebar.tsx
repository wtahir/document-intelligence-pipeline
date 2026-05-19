import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard,
  Play,
  FileSearch,
  MessageSquare,
  BarChart3,
  Shield,
  Home,
} from 'lucide-react'

const NAV = [
  { to: '/dashboard',          label: 'Overview',          icon: LayoutDashboard },
  { to: '/dashboard/pipeline',  label: 'Pipeline Runner',   icon: Play },
  { to: '/dashboard/documents', label: 'Document Explorer', icon: FileSearch },
  { to: '/dashboard/query',     label: 'Query Interface',   icon: MessageSquare },
  { to: '/dashboard/evaluation',label: 'Evaluation',        icon: BarChart3 },
]

export default function Sidebar() {
  return (
    <aside className="flex flex-col w-60 min-h-screen bg-surface-800 border-r border-surface-700 shrink-0">
      {/* Logo */}
      <div className="flex items-center gap-3 px-6 py-5 border-b border-surface-700">
        <div className="flex items-center justify-center w-9 h-9 rounded-lg bg-brand-500/20 text-brand-400">
          <Shield size={20} />
        </div>
        <div>
          <div className="text-sm font-bold text-surface-50 leading-none">Insurance AI</div>
          <div className="text-xs text-surface-400 mt-0.5">Pipeline</div>
        </div>
      </div>

      {/* Back to landing */}
      <div className="px-3 pt-3">
        <NavLink
          to="/"
          className="flex items-center gap-3 px-3 py-2 rounded-lg text-xs font-medium text-surface-500 hover:text-surface-300 hover:bg-surface-700/40 transition-colors"
        >
          <Home size={14} />
          Back to Home
        </NavLink>
      </div>

      {/* Nav */}
      <nav className="flex-1 px-3 py-4 space-y-0.5">
        {NAV.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/dashboard'}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                isActive
                  ? 'bg-brand-500/15 text-brand-400 border border-brand-500/20'
                  : 'text-surface-400 hover:text-surface-50 hover:bg-surface-700/60'
              }`
            }
          >
            <Icon size={16} />
            {label}
          </NavLink>
        ))}
      </nav>

      {/* Footer */}
      <div className="px-5 py-4 border-t border-surface-700">
        <div className="text-xs text-surface-500 font-semibold">
          Agentic RAG Pipeline
        </div>
        <div className="text-xs text-surface-600 mt-0.5">
          KG · HyDE · Self-Critique · PII
        </div>
      </div>
    </aside>
  )
}

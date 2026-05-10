import { useEffect, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from 'recharts'
import { TrendingUp, FileText, Layers, Search, DollarSign, Award } from 'lucide-react'
import PageHeader from '../components/PageHeader'
import KpiCard from '../components/KpiCard'
import StatusBadge from '../components/StatusBadge'
import Spinner from '../components/Spinner'
import { api } from '../api/client'
import type { OverviewData, StageStatus, CostTracking } from '../types'

const COLORS = ['#6366f1', '#06b6d4', '#10b981', '#f59e0b', '#ec4899', '#8b5cf6']
const STAGE_ICONS: Record<string, React.ReactNode> = {
  ingestion:  <FileText size={16} />,
  extraction: <Search size={16} />,
  chunking:   <Layers size={16} />,
  embedding:  <TrendingUp size={16} />,
  retrieval:  <Search size={16} />,
  evaluation: <Award size={16} />,
}

function StageCard({ stage }: { stage: StageStatus }) {
  return (
    <div className="card p-5 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-brand-400">
          {STAGE_ICONS[stage.key]}
          <span className="text-sm font-semibold text-surface-50">{stage.name}</span>
        </div>
        <StatusBadge status={stage.status} />
      </div>
      <div className="text-3xl font-bold text-surface-50">
        {stage.count !== null && stage.count !== undefined ? stage.count.toLocaleString() : '—'}
      </div>
      <div className="text-xs text-surface-500">{stage.mod_time ?? 'Never run'}</div>
    </div>
  )
}

export default function Overview() {
  const [data, setData] = useState<OverviewData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.get<OverviewData>('/overview')
      .then(setData)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return (
    <div className="flex items-center justify-center h-96">
      <Spinner size={8} />
    </div>
  )

  if (error) return (
    <div className="p-8 text-red-400">Error: {error}</div>
  )

  if (!data) return null

  const { kpis, stages, doc_types, damage_types, cost_tracking } = data

  const docTypeData = Object.entries(doc_types).map(([k, v]) => ({
    name: k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
    value: v,
  }))
  const damageData = Object.entries(damage_types).map(([k, v]) => ({
    name: k.charAt(0).toUpperCase() + k.slice(1),
    value: v,
  }))

  return (
    <div>
      <PageHeader
        title="Insurance Claim Intelligence"
        subtitle="End-to-end pipeline for water, storm, and glass damage claims — ingestion, classification, extraction, retrieval, and evaluation."
        badge="Overview"
      />

      <div className="px-8 py-6 space-y-8">
        {/* KPIs */}
        <div className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-7 gap-4">
          <KpiCard label="PDFs"          value={kpis.total_pdfs}       sub="uploaded" />
          <KpiCard label="Documents"     value={kpis.total_documents}   sub="ingested" />
          <KpiCard label="Chunks"        value={kpis.total_chunks}      sub="embedded" />
          <KpiCard label="Queries"       value={kpis.total_queries}     sub="executed" />
          <KpiCard label="Payouts"       value={kpis.payout_decisions}  sub="decisions" />
          <KpiCard label="Retrieval"     value={kpis.avg_retrieval_score !== null ? kpis.avg_retrieval_score?.toFixed(2) : null} sub="avg score" accent />
          <KpiCard label="Answer"        value={kpis.avg_answer_score   !== null ? kpis.avg_answer_score?.toFixed(2) : null} sub="avg score" accent />
        </div>

        {/* Stage grid */}
        <section>
          <h2 className="text-sm font-semibold text-surface-400 uppercase tracking-wider mb-4">
            Pipeline Stages
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-4">
            {stages.map(s => <StageCard key={s.key} stage={s} />)}
          </div>
        </section>

        {/* Charts row */}
        {(docTypeData.length > 0 || damageData.length > 0) && (
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
            {docTypeData.length > 0 && (
              <div className="card p-5">
                <div className="text-xs font-semibold text-surface-400 uppercase tracking-wider mb-4">
                  By Document Type
                </div>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={docTypeData} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
                    <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={false} tickLine={false} />
                    <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={false} tickLine={false} />
                    <Tooltip
                      contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, color: '#f8fafc' }}
                      cursor={{ fill: '#334155' }}
                    />
                    <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                      {docTypeData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            {damageData.length > 0 && (
              <div className="card p-5">
                <div className="text-xs font-semibold text-surface-400 uppercase tracking-wider mb-4">
                  By Damage Type
                </div>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={damageData} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
                    <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={false} tickLine={false} />
                    <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={false} tickLine={false} />
                    <Tooltip
                      contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, color: '#f8fafc' }}
                      cursor={{ fill: '#334155' }}
                    />
                    <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                      {damageData.map((_, i) => <Cell key={i} fill={COLORS[(i + 2) % COLORS.length]} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            {cost_tracking && cost_tracking.total_cost_usd > 0 && (
              <div className="card p-5 xl:col-span-1">
                <div className="text-xs font-semibold text-surface-400 uppercase tracking-wider mb-4">
                  LLM Cost Tracking
                </div>
                <div className="space-y-3 mt-2">
                  {[
                    { label: 'Extraction',  value: cost_tracking.extraction_cost_usd,  tokens: cost_tracking.extraction_tokens },
                    { label: 'Retrieval',   value: cost_tracking.retrieval_cost_usd,   tokens: cost_tracking.retrieval_tokens },
                    { label: 'Evaluation',  value: cost_tracking.eval_cost_usd,        tokens: cost_tracking.eval_tokens },
                  ].map(({ label, value, tokens }) => (
                    <div key={label} className="flex justify-between items-center">
                      <div>
                        <span className="text-sm text-surface-400">{label}</span>
                        {tokens > 0 && (
                          <span className="text-xs text-surface-600 ml-2">{tokens.toLocaleString()} tok</span>
                        )}
                      </div>
                      <span className="text-sm font-semibold text-surface-50">${value.toFixed(4)}</span>
                    </div>
                  ))}
                  <div className="border-t border-surface-700 pt-3 flex justify-between items-center">
                    <div>
                      <span className="text-sm text-surface-400 flex items-center gap-1">
                        <DollarSign size={12} /> Total cost
                      </span>
                      <span className="text-xs text-surface-600 ml-0">{cost_tracking.total_tokens.toLocaleString()} tokens</span>
                    </div>
                    <span className="text-sm font-bold text-brand-400">
                      ${cost_tracking.total_cost_usd.toFixed(4)}
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

import { useEffect, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend,
  ScatterChart, Scatter, ZAxis, Cell,
} from 'recharts'
import { ChevronDown, ChevronRight } from 'lucide-react'
import PageHeader from '../components/PageHeader'
import KpiCard from '../components/KpiCard'
import Spinner from '../components/Spinner'
import EmptyState from '../components/EmptyState'
import { api } from '../api/client'
import type { EvaluationData, EvalRow } from '../types'

function scoreColor(score: number | null, max = 5) {
  if (score === null) return 'text-surface-500'
  const pct = score / max
  return pct >= 0.7 ? 'text-emerald-400' : pct >= 0.4 ? 'text-amber-400' : 'text-red-400'
}

function ScoreBar({ value, max = 5 }: { value: number | null; max?: number }) {
  if (value === null) return <span className="text-surface-600 text-sm">—</span>
  const pct = Math.min((value / max) * 100, 100)
  const color = pct >= 70 ? 'bg-emerald-500' : pct >= 40 ? 'bg-amber-500' : 'bg-red-500'
  return (
    <div className="flex items-center gap-2 min-w-32">
      <div className="flex-1 h-1.5 bg-surface-700 rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className={`text-xs font-semibold tabular-nums ${scoreColor(value, max)}`}>
        {value.toFixed(1)}
      </span>
    </div>
  )
}

function FailurePill({ type }: { type: string | null }) {
  if (!type || type === 'none') return <span className="text-xs text-emerald-500 font-semibold">✓ none</span>
  const color = type === 'generation' ? 'text-amber-400' :
                type === 'retrieval'  ? 'text-red-400' : 'text-surface-400'
  return <span className={`text-xs font-semibold ${color}`}>{type}</span>
}

function QueryDrilldown({ row }: { row: EvalRow }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="card overflow-hidden">
      <button
        className="w-full flex items-center justify-between px-4 py-3 text-left hover:bg-surface-700/40 transition-colors"
        onClick={() => setOpen(v => !v)}
      >
        <div className="flex items-center gap-3 flex-1 min-w-0">
          {open ? <ChevronDown size={14} className="text-brand-400 shrink-0" /> : <ChevronRight size={14} className="text-surface-500 shrink-0" />}
          <span className="text-sm text-surface-200 truncate flex-1">{row.query}</span>
        </div>
        <div className="flex items-center gap-4 shrink-0 ml-4">
          <ScoreBar value={row.retrieval_score} />
          <ScoreBar value={row.answer_score} />
          <FailurePill type={row.failure_type} />
          {row.avg_distance != null && (
            <span className="text-xs text-surface-500">dist: {row.avg_distance.toFixed(3)}</span>
          )}
        </div>
      </button>

      {open && (
        <div className="px-4 pb-5 border-t border-surface-700 pt-4 space-y-4">
          {row.answer && (
            <div>
              <div className="text-xs font-semibold text-surface-400 uppercase tracking-wider mb-2">Generated Answer</div>
              <div className="text-sm text-surface-300 leading-relaxed bg-surface-800 rounded-lg p-3 whitespace-pre-wrap">{row.answer}</div>
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {row.retrieval_notes && (
              <div>
                <div className="text-xs font-semibold text-indigo-400 uppercase tracking-wider mb-1">Retrieval notes</div>
                <div className="text-xs text-surface-400 leading-relaxed">{row.retrieval_notes}</div>
              </div>
            )}
            {row.answer_notes && (
              <div>
                <div className="text-xs font-semibold text-emerald-400 uppercase tracking-wider mb-1">Answer notes</div>
                <div className="text-xs text-surface-400 leading-relaxed">{row.answer_notes}</div>
              </div>
            )}
          </div>

          {row.improvement && (
            <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg p-3">
              <div className="text-xs font-semibold text-amber-400 uppercase tracking-wider mb-1">Improvement suggestion</div>
              <div className="text-xs text-surface-300 leading-relaxed">{row.improvement}</div>
            </div>
          )}

          {row.chunks && row.chunks.length > 0 && (
            <div>
              <div className="text-xs font-semibold text-surface-400 uppercase tracking-wider mb-2">
                Retrieved Chunks ({row.chunks.length})
              </div>
              <div className="space-y-2">
                {row.chunks.map((c, ci) => (
                  <div key={ci} className="bg-surface-800 rounded-lg p-3">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs text-surface-500 font-mono">
                        {String(c.metadata?.file_name ?? `chunk ${ci + 1}`)}
                      </span>
                      <span className="text-xs text-surface-600">dist: {c.distance.toFixed(4)}</span>
                    </div>
                    <div className="text-xs text-surface-300 leading-relaxed line-clamp-4">{c.text}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {row.cost_usd > 0 && (
            <div className="text-xs text-surface-600">Query cost: ${row.cost_usd.toFixed(5)}</div>
          )}
        </div>
      )}
    </div>
  )
}

export default function Evaluation() {
  const [data, setData] = useState<EvaluationData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [sortKey, setSortKey] = useState<'retrieval_score' | 'answer_score' | 'avg_distance'>('retrieval_score')
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc')

  useEffect(() => {
    api.get<EvaluationData>('/evaluation')
      .then(setData)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div className="flex items-center justify-center h-96"><Spinner size={8} /></div>
  if (error) return <div className="p-8 text-red-400">Error: {error}</div>
  if (!data || (!data.summary && data.rows.length === 0)) {
    return (
      <div>
        <PageHeader title="Evaluation Dashboard" badge="Quality Analytics" />
        <div className="px-8 py-6">
          <EmptyState
            title="No evaluation data"
            description="Run Stage 5 (Retrieval) and Stage 6 (Evaluation) first."
          />
        </div>
      </div>
    )
  }

  const { summary, rows } = data
  const avg_ret = summary ? Number(summary.avg_retrieval_score) : null
  const avg_ans = summary ? Number(summary.avg_answer_score)    : null

  const sorted = [...rows].sort((a, b) => {
    const av = (a[sortKey] as number | null) ?? (sortDir === 'desc' ? -Infinity : Infinity)
    const bv = (b[sortKey] as number | null) ?? (sortDir === 'desc' ? -Infinity : Infinity)
    return sortDir === 'desc' ? bv - av : av - bv
  })

  function toggleSort(key: typeof sortKey) {
    if (sortKey === key) setSortDir(d => d === 'desc' ? 'asc' : 'desc')
    else { setSortKey(key); setSortDir('desc') }
  }

  const scoreData = rows.map((r, i) => ({
    idx: i + 1,
    retrieval: r.retrieval_score,
    answer: r.answer_score,
  }))

  const fCounts: Record<string, number> = {}
  rows.forEach(r => {
    const key = r.failure_type ?? 'none'
    fCounts[key] = (fCounts[key] || 0) + 1
  })
  const failData = Object.entries(fCounts).map(([name, count]) => ({ name, count }))
  const failures = rows.filter(r => r.failure_type && r.failure_type !== 'none')

  const distData = rows.flatMap((r, qi) =>
    (r.chunks || []).map(c => ({ query: qi + 1, distance: c.distance }))
  )

  const totalQueryCost = rows.reduce((s, r) => s + (r.cost_usd || 0), 0)

  const gtMetrics = summary && (summary as Record<string, unknown>).ground_truth_metrics
    ? (summary as Record<string, Record<string, number>>).ground_truth_metrics
    : null

  return (
    <div>
      <PageHeader
        title="Evaluation Dashboard"
        subtitle="GPT-as-judge evaluation with ground-truth MRR/Recall/Precision metrics. Verifies retrieval AND answer quality independently."
        badge="Quality Analytics"
      />

      <div className="px-8 py-6 space-y-8">
        {/* KPIs */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          <KpiCard label="Queries Evaluated" value={summary ? String(summary.total_queries_evaluated ?? rows.length) : rows.length} />
          <KpiCard label="Avg Retrieval" value={avg_ret !== null ? avg_ret.toFixed(2) : null} sub="out of 5" accent />
          <KpiCard label="Avg Answer"    value={avg_ans !== null ? avg_ans.toFixed(2) : null} sub="out of 5" accent />
          <KpiCard label="Failures"      value={failures.length} sub={`${rows.length > 0 ? ((failures.length / rows.length) * 100).toFixed(0) : 0}%`} />
          <KpiCard label="Eval Cost" value={totalQueryCost > 0 ? `$${totalQueryCost.toFixed(4)}` : null} sub="generation + eval" accent />
        </div>

        {/* Score + Failure charts */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          {scoreData.length > 0 && (
            <div className="card p-5">
              <div className="text-xs font-semibold text-surface-400 uppercase tracking-wider mb-4">
                Scores per Query
              </div>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={scoreData} barGap={2} margin={{ left: -20 }}>
                  <XAxis dataKey="idx" tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={false} tickLine={false} />
                  <YAxis domain={[0, 5]} tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, color: '#f8fafc' }} cursor={{ fill: '#334155' }} />
                  <Legend wrapperStyle={{ fontSize: 12, color: '#94a3b8' }} />
                  <Bar dataKey="retrieval" fill="#6366f1" radius={[3,3,0,0]} name="Retrieval" />
                  <Bar dataKey="answer"    fill="#10b981" radius={[3,3,0,0]} name="Answer" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {failData.length > 0 && (
            <div className="card p-5">
              <div className="text-xs font-semibold text-surface-400 uppercase tracking-wider mb-4">
                Failure Type Breakdown
              </div>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={failData} layout="vertical" margin={{ left: 0, right: 8 }}>
                  <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={false} tickLine={false} />
                  <YAxis dataKey="name" type="category" width={110} tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, color: '#f8fafc' }} />
                  <Bar dataKey="count" radius={[0,3,3,0]} name="Count">
                    {failData.map((entry, i) => (
                      <Cell key={i} fill={entry.name === 'none' ? '#10b981' : entry.name === 'generation' ? '#f59e0b' : '#ef4444'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>

        {/* Distance scatter */}
        {distData.length > 0 && (
          <div className="card p-5">
            <div className="text-xs font-semibold text-surface-400 uppercase tracking-wider mb-1">
              Retrieval Distance Distribution
            </div>
            <div className="text-xs text-surface-600 mb-4">
              Each dot = one retrieved chunk. Lower distance = closer semantic match. Good chunks are typically &lt; 0.4.
            </div>
            <ResponsiveContainer width="100%" height={180}>
              <ScatterChart margin={{ left: -20, right: 8, bottom: 16 }}>
                <XAxis dataKey="query" name="Query #" type="number" domain={[0, rows.length + 1]}
                  tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={false} tickLine={false}
                  label={{ value: 'Query #', fill: '#64748b', fontSize: 10, position: 'insideBottom', offset: -10 }} />
                <YAxis dataKey="distance" name="Distance" domain={[0, 1]}
                  tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={false} />
                <ZAxis range={[25, 25]} />
                <Tooltip
                  contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, color: '#f8fafc', fontSize: 12 }}
                  formatter={(val: number) => val.toFixed(4)}
                />
                <Scatter data={distData} fill="#6366f1" opacity={0.55} />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Ground truth metrics */}
        {gtMetrics && (
          <div className="card p-5">
            <div className="text-xs font-semibold text-surface-400 uppercase tracking-wider mb-4">
              Ground Truth Metrics
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {Object.entries(gtMetrics)
                .filter(([k]) => k !== 'queries_with_gt')
                .map(([k, v]) => (
                  <div key={k} className="text-center">
                    <div className="text-2xl font-bold text-brand-400">{typeof v === 'number' ? v.toFixed(3) : v}</div>
                    <div className="text-xs text-surface-500 mt-1">{k.replace(/_/g, ' ').toUpperCase()}</div>
                  </div>
                ))}
            </div>
          </div>
        )}

        {/* Per-query drill-downs */}
        <div>
          <div className="flex items-center justify-between mb-3">
            <div className="text-xs font-semibold text-surface-400 uppercase tracking-wider">
              Per-Query Results — click to expand ({rows.length})
            </div>
            <div className="flex items-center gap-1 text-xs text-surface-500">
              <span>Sort:</span>
              {(['retrieval_score', 'answer_score', 'avg_distance'] as const).map(k => (
                <button
                  key={k}
                  onClick={() => toggleSort(k)}
                  className={`px-2 py-0.5 rounded transition-colors ${sortKey === k ? 'bg-brand-600 text-white' : 'hover:text-surface-200'}`}
                >
                  {k === 'retrieval_score' ? 'Retrieval' : k === 'answer_score' ? 'Answer' : 'Distance'}
                  {sortKey === k && (sortDir === 'desc' ? ' ↓' : ' ↑')}
                </button>
              ))}
            </div>
          </div>
          <div className="space-y-2">
            {sorted.map((row, i) => (
              <QueryDrilldown key={i} row={row} />
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

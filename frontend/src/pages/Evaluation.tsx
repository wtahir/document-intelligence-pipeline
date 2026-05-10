import { useEffect, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend,
} from 'recharts'
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

export default function Evaluation() {
  const [data, setData] = useState<EvaluationData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [sortKey, setSortKey] = useState<keyof EvalRow>('retrieval_score')
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

  // Sort rows
  const sorted = [...rows].sort((a, b) => {
    const av = a[sortKey] as number | null ?? -Infinity
    const bv = b[sortKey] as number | null ?? -Infinity
    return sortDir === 'desc' ? bv - av : av - bv
  })

  function toggleSort(key: keyof EvalRow) {
    if (sortKey === key) setSortDir(d => d === 'desc' ? 'asc' : 'desc')
    else { setSortKey(key); setSortDir('desc') }
  }

  // Chart data
  const scoreData = rows.map((r, i) => ({
    idx: i + 1,
    retrieval: r.retrieval_score,
    answer: r.answer_score,
  }))

  const failures = rows.filter(r => r.failure_reason)
  const fCounts: Record<string, number> = {}
  failures.forEach(r => {
    const key = r.failure_reason!.split(':')[0].trim()
    fCounts[key] = (fCounts[key] || 0) + 1
  })
  const failData = Object.entries(fCounts).map(([name, count]) => ({ name, count }))

  return (
    <div>
      <PageHeader
        title="Evaluation Dashboard"
        subtitle="RAG pipeline quality analytics — retrieval scores, answer scores, failure breakdown, and improvement suggestions."
        badge="Quality Analytics"
      />

      <div className="px-8 py-6 space-y-8">
        {/* KPIs */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <KpiCard label="Queries Evaluated" value={summary ? String(summary.total_queries_evaluated ?? rows.length) : rows.length} />
          <KpiCard label="Avg Retrieval" value={avg_ret !== null ? avg_ret.toFixed(2) : null} sub="out of 5" accent />
          <KpiCard label="Avg Answer"    value={avg_ans !== null ? avg_ans.toFixed(2) : null} sub="out of 5" accent />
          <KpiCard label="Failures"      value={failures.length} sub={`${rows.length > 0 ? ((failures.length / rows.length) * 100).toFixed(0) : 0}%`} />
        </div>

        {/* Charts */}
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
                  <Tooltip
                    contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, color: '#f8fafc' }}
                    cursor={{ fill: '#334155' }}
                  />
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
                Failure Reasons
              </div>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={failData} layout="vertical" margin={{ left: 0, right: 8 }}>
                  <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={false} tickLine={false} />
                  <YAxis dataKey="name" type="category" width={120} tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, color: '#f8fafc' }} />
                  <Bar dataKey="count" fill="#f59e0b" radius={[0,3,3,0]} name="Count" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>

        {/* Table */}
        <div>
          <div className="text-xs font-semibold text-surface-400 uppercase tracking-wider mb-3">
            Per-Query Results ({rows.length})
          </div>
          <div className="card overflow-x-auto">
            <table className="w-full text-left">
              <thead>
                <tr className="border-b border-surface-700">
                  <th className="px-4 py-3 text-xs text-surface-400 uppercase tracking-wider">Query</th>
                  {(['retrieval_score', 'answer_score'] as const).map(k => (
                    <th
                      key={k}
                      className="px-4 py-3 text-xs text-surface-400 uppercase tracking-wider cursor-pointer hover:text-surface-50"
                      onClick={() => toggleSort(k)}
                    >
                      {k === 'retrieval_score' ? 'Retrieval' : 'Answer'}
                      {sortKey === k && (sortDir === 'desc' ? ' ↓' : ' ↑')}
                    </th>
                  ))}
                  <th className="px-4 py-3 text-xs text-surface-400 uppercase tracking-wider">Chunks</th>
                  <th className="px-4 py-3 text-xs text-surface-400 uppercase tracking-wider">Failure</th>
                  <th className="px-4 py-3 text-xs text-surface-400 uppercase tracking-wider">Suggestion</th>
                </tr>
              </thead>
              <tbody>
                {sorted.map((row, i) => (
                  <tr key={i} className="table-row">
                    <td className="px-4 py-3 text-sm text-surface-200 max-w-xs">
                      <div className="line-clamp-2">{row.query}</div>
                    </td>
                    <td className="px-4 py-3"><ScoreBar value={row.retrieval_score} /></td>
                    <td className="px-4 py-3"><ScoreBar value={row.answer_score} /></td>
                    <td className="px-4 py-3 text-sm text-surface-400">{row.chunks_used}</td>
                    <td className="px-4 py-3 text-xs text-red-400 max-w-xs">
                      {row.failure_reason ?? <span className="text-surface-600">—</span>}
                    </td>
                    <td className="px-4 py-3 text-xs text-surface-500 max-w-sm">
                      <div className="line-clamp-2">{row.improvement ?? '—'}</div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  )
}

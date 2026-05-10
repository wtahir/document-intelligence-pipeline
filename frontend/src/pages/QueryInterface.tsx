import { useState, useEffect } from 'react'
import { Send, ChevronDown, ChevronRight, Loader } from 'lucide-react'
import PageHeader from '../components/PageHeader'
import Spinner from '../components/Spinner'
import EmptyState from '../components/EmptyState'
import { api } from '../api/client'
import type { QueryResponse, QueryLogEntry } from '../types'

function ScoreDot({ score }: { score: number }) {
  const pct = Math.min(Math.max(score, 0), 1)
  const color =
    pct >= 0.7 ? 'bg-emerald-500' :
    pct >= 0.4 ? 'bg-amber-500' :
                 'bg-red-500'
  return (
    <div className="flex items-center gap-2">
      <div className={`w-2 h-2 rounded-full ${color}`} />
      <span className="text-sm font-semibold">{(pct * 5).toFixed(1)}</span>
    </div>
  )
}

function ChunkCard({ chunk, idx }: { chunk: { text: string; metadata: Record<string, unknown>; distance: number; score: number | null }; idx: number }) {
  const [open, setOpen] = useState(idx === 0)
  return (
    <div className="card overflow-hidden">
      <button
        className="w-full flex items-center justify-between px-4 py-3 text-left hover:bg-surface-700/40 transition-colors"
        onClick={() => setOpen(v => !v)}
      >
        <div className="flex items-center gap-3">
          {open ? <ChevronDown size={14} className="text-brand-400" /> : <ChevronRight size={14} className="text-surface-500" />}
          <span className="text-xs font-semibold text-surface-400">Chunk {idx + 1}</span>
          {Boolean(chunk.metadata.file_name) && (
            <span className="text-xs text-surface-500 font-mono">{String(chunk.metadata.file_name)}</span>
          )}
        </div>
        <div className="flex items-center gap-4">
          {chunk.score !== null && <ScoreDot score={chunk.score} />}
          <span className="text-xs text-surface-500">dist: {chunk.distance.toFixed(4)}</span>
        </div>
      </button>
      {open && (
        <div className="px-4 pb-4 text-sm text-surface-300 leading-relaxed border-t border-surface-700 pt-3 whitespace-pre-wrap">
          {chunk.text}
        </div>
      )}
    </div>
  )
}

export default function QueryInterface() {
  const [query, setQuery] = useState('')
  const [nResults, setNResults] = useState(5)
  const [damageFilter, setDamageFilter] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<QueryResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [logLoading, setLogLoading] = useState(true)
  const [log, setLog] = useState<QueryLogEntry[]>([])

  useEffect(() => {
    api.get<{ queries: QueryLogEntry[] }>('/query/log')
      .then(d => setLog(d.queries))
      .finally(() => setLogLoading(false))
  }, [])

  async function submit() {
    if (!query.trim()) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await api.post<QueryResponse>('/query', {
        query: query.trim(),
        n_results: nResults,
        damage_type: damageFilter || undefined,
      })
      setResult(res)
    } catch (e: unknown) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <PageHeader
        title="Query Interface"
        subtitle="Ask natural-language questions about water, storm, and glass damage claims via RAG retrieval."
        badge="AI Assistant"
      />

      <div className="px-8 py-6 grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Left: Input + Results */}
        <div className="xl:col-span-2 space-y-6">
          {/* Input card */}
          <div className="card p-5 space-y-4">
            <div>
              <div className="label">Your question</div>
              <textarea
                className="input resize-none h-28"
                placeholder="e.g. What objects were damaged in water damage claims?&#10;Show me all storm invoices and their total amounts.&#10;Which glass damage claims have severe assessments?"
                value={query}
                onChange={e => setQuery(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter' && e.ctrlKey) submit() }}
              />
            </div>

            <div className="flex gap-4 items-end flex-wrap">
              <div className="min-w-32">
                <div className="label">Chunks to retrieve</div>
                <select className="select" value={nResults} onChange={e => setNResults(Number(e.target.value))}>
                  {[3, 5, 8, 10, 15].map(n => <option key={n} value={n}>{n}</option>)}
                </select>
              </div>
              <div className="min-w-36">
                <div className="label">Damage type filter</div>
                <select className="select" value={damageFilter} onChange={e => setDamageFilter(e.target.value)}>
                  <option value="">All types</option>
                  <option value="water">Water</option>
                  <option value="storm">Storm</option>
                  <option value="glass">Glass</option>
                </select>
              </div>
              <button
                className="btn-primary ml-auto"
                onClick={submit}
                disabled={loading || !query.trim()}
              >
                {loading ? <Loader size={14} className="animate-spin" /> : <Send size={14} />}
                {loading ? 'Querying…' : 'Submit'}
              </button>
            </div>
            <div className="text-xs text-surface-600">Tip: Ctrl+Enter to submit</div>
          </div>

          {/* Error */}
          {error && (
            <div className="card p-4 border-red-500/30 bg-red-500/5 text-red-400 text-sm">
              {error}
            </div>
          )}

          {/* Answer */}
          {result && (
            <div className="space-y-4">
              {result._demo && (
                <div className="card px-4 py-2.5 border-amber-500/30 bg-amber-500/5 text-amber-400 text-xs">
                  Demo mode — showing closest pre-computed result for: <em>"{result._matched_query}"</em>
                </div>
              )}
              <div className="card p-5">
                <div className="text-xs font-semibold text-brand-400 uppercase tracking-wider mb-3">Answer</div>
                <div className="text-sm text-surface-100 leading-relaxed whitespace-pre-wrap">
                  {result.answer}
                </div>
              </div>

              <div>
                <div className="text-xs font-semibold text-surface-400 uppercase tracking-wider mb-3">
                  Retrieved Chunks ({result.chunks.length})
                </div>
                <div className="space-y-2">
                  {result.chunks.map((c, i) => (
                    <ChunkCard key={i} chunk={c} idx={i} />
                  ))}
                </div>
              </div>
            </div>
          )}

          {!result && !loading && !error && (
            <EmptyState title="No query yet" description="Type a question above and hit Submit." />
          )}
        </div>

        {/* Right: Query log */}
        <div className="space-y-4">
          <div className="text-xs font-semibold text-surface-400 uppercase tracking-wider">
            Recent Queries
          </div>
          {logLoading ? (
            <div className="flex justify-center py-8"><Spinner /></div>
          ) : log.length === 0 ? (
            <EmptyState title="No queries yet" />
          ) : (
            <div className="space-y-2 max-h-[70vh] overflow-y-auto pr-1">
              {log.map((entry, i) => (
                <button
                  key={i}
                  className="card p-4 w-full text-left hover:border-brand-500/40 transition-colors group"
                  onClick={() => setQuery(entry.query)}
                >
                  <div className="text-sm text-surface-200 group-hover:text-surface-50 line-clamp-2">
                    {entry.query}
                  </div>
                  {entry.timestamp && (
                    <div className="mt-2 text-xs text-surface-600">{entry.timestamp}</div>
                  )}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

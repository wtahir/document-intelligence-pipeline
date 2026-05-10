import { useEffect, useState, useCallback } from 'react'
import { ChevronDown, ChevronRight, Search, X } from 'lucide-react'
import PageHeader from '../components/PageHeader'
import KpiCard from '../components/KpiCard'
import StatusBadge from '../components/StatusBadge'
import Spinner from '../components/Spinner'
import EmptyState from '../components/EmptyState'
import { api } from '../api/client'
import type { Document, DocumentsResponse } from '../types'

interface Meta {
  doc_types: string[]
  damage_types: string[]
  statuses: string[]
}

function DetailRow({ label, value }: { label: string; value: React.ReactNode }) {
  if (!value && value !== 0) return null
  return (
    <div className="flex gap-2 text-sm">
      <span className="text-surface-500 shrink-0 w-36">{label}</span>
      <span className="text-surface-50 break-words">{value}</span>
    </div>
  )
}

function DocRow({ doc, expanded, onToggle }: {
  doc: Document
  expanded: boolean
  onToggle: () => void
}) {
  return (
    <>
      <tr className="table-row cursor-pointer select-none" onClick={onToggle}>
        <td className="px-4 py-3 text-sm font-mono text-surface-300 max-w-xs truncate">
          <div className="flex items-center gap-2">
            {expanded ? <ChevronDown size={14} className="text-brand-400" /> : <ChevronRight size={14} className="text-surface-500" />}
            {doc.file_name}
          </div>
        </td>
        <td className="px-4 py-3">
          {doc.document_type
            ? <span className="text-xs font-medium text-surface-300 bg-surface-700 px-2 py-0.5 rounded">
                {doc.document_type.replace(/_/g, ' ')}
              </span>
            : '—'}
        </td>
        <td className="px-4 py-3">
          {doc.damage_type
            ? <span className={`text-xs font-semibold px-2 py-0.5 rounded ${
                doc.damage_type === 'water' ? 'bg-blue-500/15 text-blue-400' :
                doc.damage_type === 'storm' ? 'bg-amber-500/15 text-amber-400' :
                doc.damage_type === 'glass' ? 'bg-cyan-500/15 text-cyan-400' :
                'bg-surface-700 text-surface-400'
              }`}>
                {doc.damage_type}
              </span>
            : '—'}
        </td>
        <td className="px-4 py-3"><StatusBadge status={doc.status} /></td>
        <td className="px-4 py-3 text-sm text-surface-400">
          {doc.confidence !== null && doc.confidence !== undefined ? doc.confidence.toFixed(2) : '—'}
        </td>
        <td className="px-4 py-3 text-sm text-surface-500 max-w-xs truncate">
          {doc.claimant_name ?? '—'}
        </td>
        <td className="px-4 py-3 text-sm text-surface-400">
          {doc.total_amount_eur != null ? `€${doc.total_amount_eur.toFixed(2)}` : '—'}
        </td>
      </tr>
      {expanded && (
        <tr className="bg-surface-900/60">
          <td colSpan={7} className="px-8 py-5">
            <div className="space-y-2">
              {doc.summary_en && (
                <div className="mb-3 p-3 rounded-lg bg-brand-500/5 border border-brand-500/20 text-sm text-surface-200 italic">
                  {doc.summary_en}
                </div>
              )}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-1">
                <DetailRow label="Claim Number"   value={doc.claim_number} />
                <DetailRow label="Policy Number"  value={doc.policy_number} />
                <DetailRow label="Language"       value={doc.language} />
                <DetailRow label="Damage Severity" value={doc.damage_severity} />
                <DetailRow label="Extracted At"   value={doc.extracted_at} />
                {doc.token_usage && (
                  <DetailRow label="Tokens used"  value={`${doc.token_usage.total_tokens.toLocaleString()} ($${doc.token_usage.cost_usd.toFixed(4)})`} />
                )}
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  )
}

export default function DocumentExplorer() {
  const [docs, setDocs] = useState<Document[]>([])
  const [total, setTotal] = useState(0)
  const [meta, setMeta] = useState<Meta>({ doc_types: [], damage_types: [], statuses: [] })
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState('')
  const [docType, setDocType] = useState('')
  const [status, setStatus] = useState('')
  const [damageType, setDamageType] = useState('')
  const [expandedRow, setExpandedRow] = useState<string | null>(null)

  const fetchDocs = useCallback(() => {
    setLoading(true)
    const params = new URLSearchParams()
    if (search)     params.set('search', search)
    if (docType)    params.set('doc_type', docType)
    if (status)     params.set('status', status)
    if (damageType) params.set('damage_type', damageType)
    params.set('limit', '200')

    api.get<DocumentsResponse>(`/documents?${params}`)
      .then(d => { setDocs(d.documents); setTotal(d.total) })
      .finally(() => setLoading(false))
  }, [search, docType, status, damageType])

  useEffect(() => {
    api.get<Meta>('/documents/meta').then(setMeta)
  }, [])

  useEffect(() => {
    const t = setTimeout(fetchDocs, 300)
    return () => clearTimeout(t)
  }, [fetchDocs])

  const success = docs.filter(d => d.status === 'success').length
  const failed  = docs.filter(d => d.status === 'failed').length

  return (
    <div>
      <PageHeader
        title="Document Explorer"
        subtitle="Browse, search, and inspect all ingested and extracted insurance documents."
        badge="Data Explorer"
      />

      <div className="px-8 py-6 space-y-6">
        {/* KPIs */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <KpiCard label="Total"     value={total}   sub="documents" />
          <KpiCard label="Success"   value={success} sub="extracted" />
          <KpiCard label="Failed"    value={failed}  sub="" />
          <KpiCard label="Doc Types" value={meta.doc_types.length} sub="distinct" />
        </div>

        {/* Filters */}
        <div className="card p-4 flex flex-wrap gap-3 items-end">
          <div className="flex-1 min-w-48">
            <div className="label">Search</div>
            <div className="relative">
              <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-surface-500" />
              <input
                className="input pl-8"
                placeholder="File name, claim number, content…"
                value={search}
                onChange={e => setSearch(e.target.value)}
              />
              {search && (
                <button className="absolute right-2 top-1/2 -translate-y-1/2 text-surface-500 hover:text-surface-50"
                        onClick={() => setSearch('')}>
                  <X size={14} />
                </button>
              )}
            </div>
          </div>
          <div className="min-w-40">
            <div className="label">Document Type</div>
            <select className="select" value={docType} onChange={e => setDocType(e.target.value)}>
              <option value="">All</option>
              {meta.doc_types.map(t => <option key={t} value={t}>{t.replace(/_/g, ' ')}</option>)}
            </select>
          </div>
          <div className="min-w-36">
            <div className="label">Damage Type</div>
            <select className="select" value={damageType} onChange={e => setDamageType(e.target.value)}>
              <option value="">All</option>
              {meta.damage_types.map(t => <option key={t} value={t}>{t}</option>)}
            </select>
          </div>
          <div className="min-w-32">
            <div className="label">Status</div>
            <select className="select" value={status} onChange={e => setStatus(e.target.value)}>
              <option value="">All</option>
              {meta.statuses.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
        </div>

        {/* Table */}
        {loading ? (
          <div className="flex justify-center py-16"><Spinner size={8} /></div>
        ) : docs.length === 0 ? (
          <EmptyState
            title="No documents found"
            description="Run the ingestion and extraction stages, or adjust your filters."
          />
        ) : (
          <div className="card overflow-x-auto">
            <table className="w-full text-left">
              <thead>
                <tr className="border-b border-surface-700">
                  {['File', 'Type', 'Damage', 'Status', 'Confidence', 'Claimant', 'Amount'].map(h => (
                    <th key={h} className="px-4 py-3 text-xs font-semibold text-surface-400 uppercase tracking-wider">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {docs.map(d => (
                  <DocRow
                    key={d.file_name}
                    doc={d}
                    expanded={expandedRow === d.file_name}
                    onToggle={() => setExpandedRow(expandedRow === d.file_name ? null : d.file_name)}
                  />
                ))}
              </tbody>
            </table>
            <div className="px-4 py-3 text-xs text-surface-500 border-t border-surface-700">
              Showing {docs.length} of {total} documents
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

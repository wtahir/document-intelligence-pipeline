import { useEffect, useRef, useState } from 'react'
import { Play, Loader } from 'lucide-react'
import PageHeader from '../components/PageHeader'
import StatusBadge from '../components/StatusBadge'
import Spinner from '../components/Spinner'
import { api } from '../api/client'
import { useConfig } from '../context/ConfigContext'

interface StageInfo {
  key: string
  name: string
  status: string
  count: number | null
  mod_time: string | null
}

const STAGE_DESCRIPTIONS: Record<string, string> = {
  ingestion:  'Extract text from PDFs using pdfplumber',
  extraction: 'Classify and extract structured fields with GPT-4o',
  chunking:   'Split documents into overlapping chunks',
  embedding:  'Embed chunks with sentence-transformers into ChromaDB',
  retrieval:  'Execute predefined RAG queries and generate answers',
  evaluation: 'Score retrieval and answer quality with GPT-4o-as-judge',
}

interface LogLine {
  stage: string
  line?: string
  done?: boolean
  returncode?: number
  started?: boolean
  abort?: boolean
}

export default function PipelineRunner() {
  const [stages, setStages] = useState<StageInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [runningStage, setRunningStage] = useState<string | null>(null)
  const [logs, setLogs] = useState<{ stage: string; line: string }[]>([])
  const [error, setError] = useState<string | null>(null)
  const logRef = useRef<HTMLDivElement>(null)
  const { demo_mode } = useConfig()

  function fetchStatus() {
    setLoading(true)
    api.get<{ stages: StageInfo[] }>('/pipeline/status')
      .then(d => setStages(d.stages))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }

  useEffect(() => { fetchStatus() }, [])

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight
    }
  }, [logs])

  async function runStage(key: string) {
    setRunningStage(key)
    setLogs([])

    const res = await fetch(`/api/pipeline/run/${key}`, { method: 'POST' })
    if (!res.body) { setRunningStage(null); return }

    const reader = res.body.getReader()
    const decoder = new TextDecoder()

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      const text = decoder.decode(value)
      const lines = text.split('\n').filter(l => l.startsWith('data: '))
      for (const l of lines) {
        try {
          const obj = JSON.parse(l.slice(6)) as LogLine
          if (obj.line !== undefined) {
            setLogs(prev => [...prev, { stage: key, line: obj.line! }])
          }
          if (obj.done) break
        } catch { /* ignore parse errors in stream */ }
      }
    }

    setRunningStage(null)
    fetchStatus()
  }

  async function runAll() {
    setRunningStage('all')
    setLogs([])

    const res = await fetch('/api/pipeline/run-all', { method: 'POST' })
    if (!res.body) { setRunningStage(null); return }

    const reader = res.body.getReader()
    const decoder = new TextDecoder()

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      const text = decoder.decode(value)
      const lines = text.split('\n').filter(l => l.startsWith('data: '))
      for (const l of lines) {
        try {
          const obj = JSON.parse(l.slice(6)) as LogLine
          if (obj.line !== undefined) {
            setLogs(prev => [...prev, { stage: obj.stage ?? 'all', line: obj.line! }])
          }
          if (obj.abort) break
        } catch { /* ignore */ }
      }
    }

    setRunningStage(null)
    fetchStatus()
  }

  return (
    <div>
      <PageHeader
        title="Pipeline Runner"
        subtitle="Execute individual stages or the full pipeline. Logs stream in real-time."
        badge="Operations"
        actions={
          <button
            className="btn-primary"
            onClick={runAll}
            disabled={runningStage !== null || demo_mode}
            title={demo_mode ? 'Disabled in demo mode' : undefined}
          >
            {runningStage === 'all' ? <Loader size={14} className="animate-spin" /> : <Play size={14} />}
            Run All Stages
          </button>
        }
      />

      <div className="px-8 py-6 space-y-6">
        {error && <div className="text-red-400 text-sm">{error}</div>}

        {loading ? (
          <div className="flex justify-center py-16"><Spinner size={8} /></div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
            {stages.map(s => {
              const isRunning = runningStage === s.key
              return (
                <div key={s.key} className="card p-5 flex flex-col gap-4">
                  <div className="flex items-start justify-between">
                    <div>
                      <div className="font-semibold text-surface-50 capitalize">{s.name}</div>
                      <div className="text-xs text-surface-500 mt-0.5">{STAGE_DESCRIPTIONS[s.key]}</div>
                    </div>
                    <StatusBadge status={s.status} />
                  </div>

                  <div className="flex items-center justify-between text-sm">
                    <div>
                      <span className="text-2xl font-bold text-surface-50">
                        {s.count !== null && s.count !== undefined ? s.count.toLocaleString() : '—'}
                      </span>
                    </div>
                    <div className="text-xs text-surface-500">{s.mod_time ?? 'Never'}</div>
                  </div>

                  <button
                    className="btn-secondary w-full justify-center"
                    onClick={() => runStage(s.key)}
                    disabled={runningStage !== null || demo_mode}
                    title={demo_mode ? 'Disabled in demo mode' : undefined}
                  >
                    {isRunning
                      ? <><Loader size={13} className="animate-spin" /> Running…</>
                      : <><Play size={13} /> Run {s.name}</>
                    }
                  </button>
                </div>
              )
            })}
          </div>
        )}

        {/* Log console */}
        {logs.length > 0 && (
          <div>
            <div className="text-xs font-semibold text-surface-400 uppercase tracking-wider mb-2">
              Live Output
            </div>
            <div
              ref={logRef}
              className="card bg-surface-900 p-4 h-72 overflow-y-auto font-mono text-xs leading-5 text-surface-300 space-y-0.5"
            >
              {logs.map((l, i) => (
                <div key={i} className="flex gap-3">
                  <span className="text-brand-500/60 shrink-0 w-20 truncate">[{l.stage}]</span>
                  <span>{l.line}</span>
                </div>
              ))}
              {runningStage && (
                <div className="flex items-center gap-2 text-surface-500 pt-1">
                  <Spinner size={3} /> <span>Running…</span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Upload section */}
        <div className="card p-6">
          <div className="text-sm font-semibold text-surface-50 mb-3">Upload PDFs</div>
          <UploadZone onUploaded={fetchStatus} />
        </div>
      </div>
    </div>
  )
}

function UploadZone({ onUploaded }: { onUploaded: () => void }) {
  const [uploading, setUploading] = useState(false)
  const [result, setResult] = useState<string | null>(null)

  async function handleFiles(files: FileList) {
    setUploading(true)
    setResult(null)
    const form = new FormData()
    for (const f of Array.from(files)) {
      form.append('files', f)
    }
    try {
      const res = await fetch('/api/upload', { method: 'POST', body: form })
      const data = await res.json()
      setResult(`Uploaded ${data.count} file(s): ${data.saved.join(', ')}`)
      onUploaded()
    } catch (e: unknown) {
      setResult(`Error: ${(e as Error).message}`)
    } finally {
      setUploading(false)
    }
  }

  return (
    <div>
      <label
        className="flex flex-col items-center justify-center w-full h-28 border-2 border-dashed
                   border-surface-600 rounded-xl cursor-pointer hover:border-brand-500
                   hover:bg-brand-500/5 transition-colors"
      >
        <input
          type="file"
          accept=".pdf"
          multiple
          className="hidden"
          onChange={e => e.target.files && handleFiles(e.target.files)}
        />
        {uploading
          ? <Spinner size={6} />
          : (
            <div className="text-center">
              <div className="text-sm text-surface-400">Drop PDFs here or click to browse</div>
              <div className="text-xs text-surface-600 mt-1">.pdf only</div>
            </div>
          )
        }
      </label>
      {result && (
        <div className="mt-3 text-sm text-emerald-400">{result}</div>
      )}
    </div>
  )
}

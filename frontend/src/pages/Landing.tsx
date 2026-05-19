import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Shield, FileText, Brain, Search, Layers, RefreshCw,
  GitBranch, Sparkles, ArrowRight, CheckCircle2, Zap,
  Lock, DollarSign, Activity, Server, Clock, ChevronDown,
} from 'lucide-react'

/* ─── Backend warm-up status ─────────────────────────────────── */
type BackendStatus = 'cold' | 'waking' | 'ready'

function useBackendWarmup() {
  const [status, setStatus] = useState<BackendStatus>('cold')

  useEffect(() => {
    let cancelled = false
    setStatus('waking')

    const ping = async () => {
      try {
        const res = await fetch('/api/health')
        if (!cancelled && res.ok) setStatus('ready')
      } catch {
        // retry after delay
        if (!cancelled) setTimeout(ping, 3000)
      }
    }
    ping()

    return () => { cancelled = true }
  }, [])

  return status
}

/* ─── Animated counter ───────────────────────────────────────── */
function Counter({ end, suffix = '' }: { end: number; suffix?: string }) {
  const [value, setValue] = useState(0)
  useEffect(() => {
    const duration = 1600
    const steps = 40
    const increment = end / steps
    let current = 0
    const timer = setInterval(() => {
      current += increment
      if (current >= end) { setValue(end); clearInterval(timer) }
      else setValue(Math.floor(current))
    }, duration / steps)
    return () => clearInterval(timer)
  }, [end])
  return <>{value.toLocaleString()}{suffix}</>
}

/* ─── Status indicator pill ──────────────────────────────────── */
function BackendPill({ status }: { status: BackendStatus }) {
  const cfg = {
    cold:   { color: 'bg-surface-600', text: 'text-surface-400', label: 'Connecting…',  icon: <Clock size={12} /> },
    waking: { color: 'bg-amber-500/20', text: 'text-amber-400',  label: 'Warming up…', icon: <Activity size={12} className="animate-pulse" /> },
    ready:  { color: 'bg-emerald-500/20', text: 'text-emerald-400', label: 'Live',      icon: <CheckCircle2 size={12} /> },
  }[status]

  return (
    <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium ${cfg.color} ${cfg.text}`}>
      {cfg.icon} {cfg.label}
    </span>
  )
}

/* ─── Pipeline step for the flow visual ──────────────────────── */
const PIPELINE_STEPS = [
  { icon: <FileText size={20} />, title: 'Document Ingestion', desc: 'OCR + multi-format parsing of PDFs, scanned documents, and structured data across any domain', color: 'from-amber-400 to-orange-600', business: 'Digitize paper processes' },
  { icon: <Search size={20} />, title: 'Intelligent Extraction', desc: 'LLM-powered extraction of entities, relationships, dates, and domain-specific fields with confidence scores', color: 'from-rose-400 to-pink-700', business: 'Eliminate manual data entry' },
  { icon: <Layers size={20} />, title: 'Smart Chunking', desc: 'Semantic-aware document splitting with overlap for optimal retrieval context', color: 'from-purple-500 to-violet-800', business: 'Maximize AI accuracy' },
  { icon: <GitBranch size={20} />, title: 'Knowledge Graph', desc: 'Structured entity relationships linking people, documents, events, and concepts across the corpus', color: 'from-yellow-500 to-amber-700', business: 'Connect the dots automatically' },
  { icon: <Brain size={20} />, title: 'Agentic Retrieval', desc: 'Self-correcting retrieval loop with query intelligence, HyDE, and multi-strategy routing', color: 'from-red-500 to-rose-800', business: 'Answer complex questions' },
  { icon: <RefreshCw size={20} />, title: 'Self-Critique & Verification', desc: 'LLM verifies its own answers, checks citations, retries if insufficient', color: 'from-orange-400 to-amber-700', business: 'Build trust in AI outputs' },
]

const CAPABILITIES = [
  { icon: <FileText size={18} />, label: 'OCR / Document Ingestion', desc: 'Parse PDFs, scanned documents, and structured data' },
  { icon: <Sparkles size={18} />, label: 'LLM Reasoning Pipeline', desc: 'Multi-step extraction with confidence scoring' },
  { icon: <RefreshCw size={18} />, label: 'Self-Correcting Retrieval', desc: 'Iterative retrieval with automated quality checks' },
  { icon: <Lock size={18} />, label: 'PII Redaction', desc: 'Automatic detection and masking of sensitive data' },
  { icon: <Server size={18} />, label: 'Async Processing + APIs', desc: 'FastAPI + Celery with real-time status updates' },
  { icon: <DollarSign size={18} />, label: 'Cost Optimization', desc: 'Token tracking, caching, and smart model routing' },
]

const TECH = [
  'Python', 'FastAPI', 'React + TypeScript', 'ChromaDB', 'OpenAI GPT',
  'Sentence-Transformers', 'Cross-Encoder Reranking', 'BM25 Hybrid Search',
  'Knowledge Graph (NetworkX)', 'PII Redaction', 'Docker', 'Render Cloud',
  'Celery + Redis', 'Streaming SSE',
]

/* ═══════════════════════════════════════════════════════════════ */
export default function Landing() {
  const navigate = useNavigate()
  const backendStatus = useBackendWarmup()

  return (
    <div className="min-h-screen bg-surface-900 text-surface-50 overflow-x-hidden">

      {/* ── HERO ──────────────────────────────────────────────── */}
      <section className="relative flex flex-col items-center justify-center min-h-[92vh] px-6 text-center overflow-hidden">
        {/* Star field */}
        <div className="star-field" />

        {/* Nebula orbs — warm, cosmic, NOT blue */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          {/* Amber nebula — upper left */}
          <div className="absolute -top-40 -left-40 w-[600px] h-[600px] rounded-full bg-amber-500/[0.09] blur-[130px] animate-glow-pulse animate-drift" />
          {/* Crimson nebula — lower right */}
          <div className="absolute -bottom-60 -right-40 w-[700px] h-[700px] rounded-full bg-rose-700/[0.08] blur-[150px] animate-glow-pulse" style={{ animationDelay: '3s' }} />
          {/* Deep orange haze — centre */}
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[900px] h-[500px] rounded-full bg-orange-700/[0.05] blur-[110px] animate-glow-pulse" style={{ animationDelay: '1.5s' }} />
          {/* Faint magenta wisp — upper right */}
          <div className="absolute -top-20 right-0 w-[350px] h-[350px] rounded-full bg-pink-800/[0.07] blur-[100px] animate-glow-pulse" style={{ animationDelay: '5s' }} />
        </div>

        {/* Content */}
        <div className="relative z-10 max-w-4xl mx-auto">
          {/* Status bar */}
          <div className="flex items-center justify-center gap-3 mb-8 animate-fade-in" style={{ animationDelay: '0.1s' }}>
            <BackendPill status={backendStatus} />
            <span className="text-xs text-surface-500">Render (free tier — cold starts expected)</span>
          </div>

          {/* Logo */}
          <div className="flex items-center justify-center gap-3 mb-6 animate-fade-in-up" style={{ animationDelay: '0.2s' }}>
            <div className="flex items-center justify-center w-14 h-14 rounded-2xl bg-gradient-to-br from-amber-400 to-orange-700 shadow-lg shadow-amber-500/25">
              <Shield size={28} className="text-white" />
            </div>
          </div>

          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold leading-tight tracking-tight animate-fade-in-up" style={{ animationDelay: '0.3s' }}>
            Document
            <span className="block bg-gradient-to-r from-amber-300 via-orange-400 to-rose-400 bg-clip-text text-transparent">
              Intelligence Pipeline
            </span>
          </h1>

          <p className="mt-6 text-lg sm:text-xl text-surface-400 max-w-2xl mx-auto leading-relaxed animate-fade-in-up" style={{ animationDelay: '0.45s' }}>
            An <strong className="text-surface-200">enterprise-grade document AI system</strong> that ingests semi-structured
            documents, extracts entities &amp; relationships, builds knowledge graphs, and answers complex
            queries with <strong className="text-surface-200">self-correcting retrieval</strong> and <strong className="text-surface-200">LLM self-critique</strong>.
            Demo uses insurance claims as a sample domain; pipeline is <strong className="text-surface-200">domain-agnostic</strong>.
          </p>

          {/* Business value callout */}
          <div className="mt-8 inline-flex items-center gap-2 px-5 py-2.5 rounded-full bg-surface-800/80 border border-amber-800/30 text-sm text-surface-300 animate-fade-in-up" style={{ animationDelay: '0.55s' }}>
            <Zap size={14} className="text-amber-400" />
            Built for digital transformation. From raw documents to AI-powered decisions
          </div>

          {/* CTA */}
          <div className="mt-10 flex flex-col sm:flex-row items-center justify-center gap-4 animate-fade-in-up" style={{ animationDelay: '0.65s' }}>
            <button
              onClick={() => navigate('/dashboard')}
              disabled={backendStatus !== 'ready'}
              className="group relative inline-flex items-center gap-2 px-8 py-3.5 rounded-xl font-semibold text-white
                         bg-gradient-to-r from-amber-500 to-orange-600 hover:from-amber-400 hover:to-orange-500
                         shadow-lg shadow-amber-600/30 transition-all duration-200
                         disabled:opacity-50 disabled:cursor-not-allowed disabled:shadow-none"
            >
              {backendStatus === 'ready' ? (
                <>
                  Explore Live Pipeline
                  <ArrowRight size={18} className="transition-transform group-hover:translate-x-1" />
                </>
              ) : (
                <>
                  <Activity size={18} className="animate-pulse" />
                  Backend warming up…
                </>
              )}
            </button>
            <a
              href="https://github.com/wtahir/insurance-pipeline"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-6 py-3.5 rounded-xl font-semibold
                         text-surface-200 bg-surface-800/60 border border-surface-600/60 hover:bg-surface-700/60 hover:border-amber-800/40 transition-colors"
            >
              <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"/></svg>
              View Source Code
            </a>
          </div>

          {/* Quick stats - no API needed */}
          <div className="mt-16 grid grid-cols-2 sm:grid-cols-4 gap-6 max-w-2xl mx-auto animate-fade-in-up" style={{ animationDelay: '0.8s' }}>
            {[
              { value: 90, suffix: '', label: 'Documents Processed' },
              { value: 6, suffix: '', label: 'Pipeline Stages' },
              { value: 7, suffix: '', label: 'AI Components' },
              { value: 15, suffix: 'k+', label: 'Lines of Code' },
            ].map(({ value, suffix, label }) => (
              <div key={label} className="text-center">
                <div className="text-3xl font-bold text-surface-50">
                  <Counter end={value} suffix={suffix} />
                </div>
                <div className="text-xs text-surface-500 mt-1">{label}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Scroll hint */}
        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 animate-bounce">
          <div className="w-14 h-14 rounded-full bg-gradient-to-br from-amber-400 to-orange-600 shadow-lg shadow-amber-500/40 flex items-center justify-center">
            <ChevronDown size={32} strokeWidth={3.5} className="text-white drop-shadow-md" />
          </div>
        </div>
      </section>

      {/* ── BUSINESS VALUE ────────────────────────────────────── */}
      <section className="px-6 py-20 bg-surface-800/30 border-y border-amber-900/20" style={{ backgroundImage: 'radial-gradient(ellipse 80% 40% at 50% 50%, rgba(251,146,60,0.03) 0%, transparent 70%)' }}>
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-14">
            <span className="text-xs font-bold text-amber-400 uppercase tracking-widest">The Problem & Solution</span>
            <h2 className="mt-3 text-3xl sm:text-4xl font-bold">
              From Raw Documents to <span className="text-amber-400">AI-Powered Decisions</span>
            </h2>
            <p className="mt-4 text-surface-400 max-w-2xl mx-auto">
              Organisations across every industry process thousands of documents daily. This pipeline demonstrates how AI
              can automate document understanding, extract structured data, and provide verified answers,
              reducing processing time from hours to seconds. Insurance claims are used as the demo domain.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              {
                emoji: '📄',
                title: 'Manual Processing',
                problem: true,
                items: [
                  'Knowledge workers read each document manually',
                  'Hours spent on data entry and cross-referencing',
                  'Human errors in value and entity extraction',
                  'Inconsistent decision-making across teams',
                ],
              },
              {
                emoji: '🤖',
                title: 'AI Pipeline',
                problem: false,
                items: [
                  'Documents ingested and parsed automatically',
                  'LLM extracts entities with confidence scores',
                  'Knowledge graph connects related documents and entities',
                  'Self-correcting retrieval ensures accuracy',
                ],
              },
              {
                emoji: '📊',
                title: 'Business Impact',
                problem: false,
                items: [
                  '~95% reduction in processing time',
                  'Consistent, auditable AI reasoning',
                  'PII redaction for compliance',
                  'Cost tracking per operation',
                ],
              },
            ].map(({ emoji, title, problem, items }) => (
              <div key={title} className={`rounded-2xl p-6 border ${problem ? 'border-red-500/20 bg-red-500/5' : 'border-emerald-500/20 bg-emerald-500/5'}`}>
                <div className="text-3xl mb-3">{emoji}</div>
                <h3 className={`text-lg font-bold mb-4 ${problem ? 'text-red-400' : 'text-emerald-400'}`}>{title}</h3>
                <ul className="space-y-2">
                  {items.map(item => (
                    <li key={item} className="flex items-start gap-2 text-sm text-surface-300">
                      <span className={`mt-1 shrink-0 ${problem ? 'text-red-500' : 'text-emerald-500'}`}>
                        {problem ? '✕' : '✓'}
                      </span>
                      {item}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── PIPELINE FLOW ─────────────────────────────────────── */}
      <section className="px-6 py-20" style={{ backgroundImage: 'radial-gradient(ellipse 60% 50% at 50% 0%, rgba(251,191,36,0.03) 0%, transparent 70%)' }}>
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-14">
            <span className="text-xs font-bold text-amber-400 uppercase tracking-widest">How It Works</span>
            <h2 className="mt-3 text-3xl sm:text-4xl font-bold">
              End-to-End <span className="text-amber-400">Processing Pipeline</span>
            </h2>
            <p className="mt-4 text-surface-400 max-w-2xl mx-auto">
              Six interconnected stages transform raw documents, from any domain, into queryable intelligence.
              Each stage is independently runnable, observable, and produces structured output.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {PIPELINE_STEPS.map((step, i) => (
              <div key={i} className="group relative rounded-2xl bg-surface-800 border border-surface-700 p-6 hover:border-surface-600 transition-all duration-300">
                {/* Step number */}
                <div className="absolute -top-3 -left-2 w-7 h-7 rounded-full bg-surface-900 border border-surface-700 flex items-center justify-center text-xs font-bold text-surface-400">
                  {i + 1}
                </div>

                <div className={`inline-flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-br ${step.color} text-white mb-4`}>
                  {step.icon}
                </div>
                <h3 className="text-base font-bold text-surface-50 mb-2">{step.title}</h3>
                <p className="text-sm text-surface-400 leading-relaxed mb-3">{step.desc}</p>
                <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-amber-500/10 text-amber-400 text-xs font-medium">
                  <Zap size={10} />
                  {step.business}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── ENTERPRISE CAPABILITIES ───────────────────────────── */}
      <section className="px-6 py-20 bg-surface-800/25 border-y border-rose-900/20" style={{ backgroundImage: 'radial-gradient(ellipse 70% 50% at 70% 50%, rgba(225,29,72,0.04) 0%, transparent 70%)' }}>
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-14">
            <span className="text-xs font-bold text-amber-400 uppercase tracking-widest">Enterprise AI Signals</span>
            <h2 className="mt-3 text-3xl sm:text-4xl font-bold">
              Production-Grade <span className="text-amber-400">Capabilities</span>
            </h2>
            <p className="mt-4 text-surface-400 max-w-2xl mx-auto">
              Built with the patterns and practices required for real enterprise AI deployments.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
            {CAPABILITIES.map(({ icon, label, desc }) => (
              <div key={label} className="flex items-start gap-4 rounded-xl bg-surface-800 border border-surface-700 p-5">
                <div className="shrink-0 w-10 h-10 rounded-lg bg-amber-500/10 text-amber-400 flex items-center justify-center">
                  {icon}
                </div>
                <div>
                  <div className="text-sm font-semibold text-surface-100">{label}</div>
                  <div className="text-xs text-surface-400 mt-1 leading-relaxed">{desc}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── WHAT MAKES IT DIFFERENT ───────────────────────────── */}
      <section className="px-6 py-20" style={{ backgroundImage: 'radial-gradient(ellipse 60% 50% at 30% 80%, rgba(251,146,60,0.04) 0%, transparent 70%)' }}>
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-14">
            <span className="text-xs font-bold text-amber-400 uppercase tracking-widest">Technical Depth</span>
            <h2 className="mt-3 text-3xl sm:text-4xl font-bold">
              Beyond Basic <span className="text-amber-400">Vector RAG</span>
            </h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="rounded-2xl border border-red-500/20 bg-red-500/5 p-6">
              <div className="text-sm font-bold text-red-400 uppercase tracking-wider mb-4">
                Typical RAG Project
              </div>
              <ul className="space-y-2.5 text-sm text-surface-400">
                {[
                  'Embed raw query → retrieve top-k → generate',
                  'Single-shot retrieval, no error correction',
                  'No query understanding or routing',
                  'Breaks on complex multi-hop questions',
                  'No structured knowledge, only vectors',
                  'Hallucination-prone with no verification',
                ].map(item => (
                  <li key={item} className="flex items-start gap-2">
                    <span className="text-red-500 mt-0.5 shrink-0">✕</span>
                    {item}
                  </li>
                ))}
              </ul>
            </div>
            <div className="rounded-2xl border border-emerald-500/20 bg-emerald-500/5 p-6">
              <div className="text-sm font-bold text-emerald-400 uppercase tracking-wider mb-4">
                This Pipeline
              </div>
              <ul className="space-y-2.5 text-sm text-surface-400">
                {[
                  'Query intelligence routes to optimal strategy',
                  'Self-correcting retrieval loop (max 3 iterations)',
                  'HyDE + multi-query for semantic gap bridging',
                  'Knowledge graph for verified entity lookups',
                  'Context engineering: dedup, compress, organize',
                  'LLM self-critique with citation verification',
                ].map(item => (
                  <li key={item} className="flex items-start gap-2">
                    <span className="text-emerald-500 mt-0.5 shrink-0">✓</span>
                    {item}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* ── TECH STACK ────────────────────────────────────────── */}
      <section className="px-6 py-16 bg-surface-800/25 border-y border-amber-900/20">
        <div className="max-w-4xl mx-auto text-center">
          <span className="text-xs font-bold text-amber-400 uppercase tracking-widest">Technology Stack</span>
          <div className="mt-6 flex flex-wrap items-center justify-center gap-2">
            {TECH.map(t => (
              <span key={t} className="px-3 py-1.5 rounded-full text-xs font-medium bg-surface-700/40 text-surface-300 border border-surface-600/50 hover:border-amber-800/50 transition-colors">
                {t}
              </span>
            ))}
          </div>
        </div>
      </section>

      {/* ── FINAL CTA ─────────────────────────────────────────── */}
      <section className="px-6 py-24 text-center" style={{ backgroundImage: 'radial-gradient(ellipse 60% 60% at 50% 50%, rgba(251,191,36,0.04) 0%, transparent 70%)' }}>
        <div className="max-w-2xl mx-auto">
          <h2 className="text-3xl font-bold mb-4">Ready to Explore?</h2>
          <p className="text-surface-400 mb-8">
            See the live pipeline dashboard with real processed data, interactive queries, and detailed evaluation metrics.
          </p>
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <button
              onClick={() => navigate('/dashboard')}
              disabled={backendStatus !== 'ready'}
              className="group inline-flex items-center gap-2 px-8 py-3.5 rounded-xl font-semibold text-white
                         bg-gradient-to-r from-amber-500 to-orange-600 hover:from-amber-400 hover:to-orange-500
                         shadow-lg shadow-amber-600/30 transition-all duration-200
                         disabled:opacity-50 disabled:cursor-not-allowed disabled:shadow-none"
            >
              {backendStatus === 'ready' ? (
                <>
                  Open Pipeline Dashboard
                  <ArrowRight size={18} className="transition-transform group-hover:translate-x-1" />
                </>
              ) : (
                <>
                  <Activity size={18} className="animate-pulse" />
                  Waiting for backend…
                </>
              )}
            </button>
            <BackendPill status={backendStatus} />
          </div>
          {backendStatus !== 'ready' && (
            <p className="mt-4 text-xs text-surface-600">
              The backend runs on Render's free tier and needs ~30 seconds to wake up on first visit. Hang tight!
            </p>
          )}
        </div>
      </section>

      {/* ── FOOTER ────────────────────────────────────────────── */}
      <footer className="border-t border-amber-900/20 px-6 py-8 text-center">
        <div className="text-xs text-surface-600">
          Built by <a href="https://github.com/wtahir" target="_blank" rel="noopener noreferrer" className="text-amber-400 hover:underline">Waqas Tahir</a>
          {' · '}
          <a href="https://github.com/wtahir/insurance-pipeline" target="_blank" rel="noopener noreferrer" className="text-surface-400 hover:text-surface-200 transition-colors">GitHub</a>
        </div>
      </footer>
    </div>
  )
}

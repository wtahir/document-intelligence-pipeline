// Shared TypeScript types for the Insurance AI Pipeline UI

export interface StageStatus {
  key: string
  name: string
  script: string
  status: 'not_run' | 'complete' | 'partial'
  count: number | null
  mod_time: string | null
}

export interface OverviewKpis {
  total_pdfs: number
  total_documents: number
  total_chunks: number
  total_queries: number
  avg_retrieval_score: number | null
  avg_answer_score: number | null
  payout_decisions: number
}

export interface OverviewData {
  stages: StageStatus[]
  kpis: OverviewKpis
  doc_types: Record<string, number>
  damage_types: Record<string, number>
  token_usage: Record<string, number>
  summaries: Record<string, unknown>
}

export interface Document {
  file_name: string
  file_path?: string
  status: 'success' | 'failed' | 'skipped'
  document_type?: string
  damage_type?: string
  language?: string
  claimant_name?: string | null
  claim_number?: string | null
  policy_number?: string | null
  confidence?: number | null
  summary_en?: string | null
  total_amount_eur?: number | null
  damage_severity?: string | null
  extracted_at?: string
  original_content?: string
  content?: string
  token_usage?: {
    prompt_tokens: number
    completion_tokens: number
    total_tokens: number
    cost_usd: number
  }
}

export interface DocumentsResponse {
  total: number
  documents: Document[]
}

export interface QueryChunk {
  text: string
  metadata: Record<string, unknown>
  distance: number
  score: number | null
}

export interface QueryResponse {
  query: string
  answer: string
  chunks: QueryChunk[]
  filters: Record<string, string>
  _demo?: boolean
  _matched_query?: string
}

export interface QueryLogEntry {
  query: string
  answer?: string
  chunks_retrieved?: number
  top_distance?: number
  timestamp?: string
}

export interface EvalRow {
  query: string
  retrieval_score: number | null
  answer_score: number | null
  chunks_used: number
  top_distance: number | null
  failure_reason: string | null
  improvement: string | null
}

export interface EvaluationData {
  summary: Record<string, unknown> | null
  rows: EvalRow[]
  total_queries_in_log: number
}

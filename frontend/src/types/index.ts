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

export interface CostTracking {
  extraction_cost_usd: number
  retrieval_cost_usd: number
  eval_cost_usd: number
  total_cost_usd: number
  extraction_tokens: number
  retrieval_tokens: number
  eval_tokens: number
  total_tokens: number
}

export interface OverviewData {
  stages: StageStatus[]
  kpis: OverviewKpis
  doc_types: Record<string, number>
  damage_types: Record<string, number>
  token_usage: Record<string, number>
  cost_tracking: CostTracking
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

export interface QueryPlan {
  complexity: string
  strategy: string
  reasoning: string
  sub_queries?: string[]
  hyde_document?: string
  entities_extracted?: string[]
}

export interface GraphFact {
  subject: string
  predicate: string
  object: string
}

export interface SelfCritique {
  quality: string
  issues: string[]
  missing_info: string
  should_retry: boolean
}

export interface ContextEngineering {
  chunks_before_dedup: number
  chunks_after_dedup: number
  compression_applied: boolean
  graph_enrichment: boolean
  hierarchical_organization: boolean
}

export interface QueryResponse {
  query: string
  answer: string
  chunks: QueryChunk[]
  filters: Record<string, string>
  pipeline?: string
  query_plan?: QueryPlan
  graph_facts?: GraphFact[]
  self_critique?: SelfCritique
  retrieval_iterations?: number
  context_engineering?: ContextEngineering
  token_usage?: Record<string, number>
  latency_seconds?: number
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
  avg_distance: number | null
  failure_type: string | null
  improvement: string | null
  answer: string
  retrieval_notes: string | null
  answer_notes: string | null
  chunks: Array<{ text: string; metadata: Record<string, unknown>; distance: number; rerank_score?: number }>
  cost_usd: number
}

export interface EvaluationData {
  summary: Record<string, unknown> | null
  rows: EvalRow[]
  total_queries_in_log: number
}

"Why did you choose sentence-transformers over OpenAI embeddings?"
"Two reasons — cost and control. OpenAI embeddings are an API call, which means every embedding operation has latency, cost, and a dependency on an external service. In a pipeline processing thousands of documents, that adds up quickly and introduces a failure point outside your control. Sentence-transformers runs locally, so embedding is free after the initial model download and works offline. The second reason is multilingual support — paraphrase-multilingual-MiniLM-L12-v2 was specifically trained on parallel multilingual data, which means German source documents and English queries land in the same vector space. OpenAI's ada-002 also handles multilingual text well, but for this use case the local model performed adequately and the operational tradeoffs favored keeping it in-process. In a production system I'd benchmark both on a held-out evaluation set before deciding."

"How would you scale this pipeline to 100,000 documents?"
"The current pipeline is sequential — it processes one document at a time. At 100,000 documents that breaks in several ways. First, the ingestion and extraction stages would need to be parallelized. I'd containerize each stage with Docker and use a message queue like RabbitMQ or Celery with Redis as the broker — documents drop into a queue and a pool of workers processes them concurrently. Second, the embedding stage is the bottleneck — sentence-transformers on CPU is slow at scale. I'd either move to GPU inference or switch to a hosted embedding endpoint for batch processing. Third, ChromaDB's local persistent client doesn't scale horizontally — I'd replace it with a production vector database like Qdrant or Weaviate that supports distributed deployment. Fourth, the LLM extraction calls in Stage 2 hit rate limits at scale — I'd implement retry logic with exponential backoff and batch documents across multiple API keys or use Azure's provisioned throughput. Finally, I'd add a job tracking database so the pipeline can resume from any failed point rather than reprocessing everything."

"How would you improve retrieval quality if evaluation scores were low?"
"I'd approach it systematically by first identifying whether it's a retrieval problem or a generation problem — which is why I built the evaluation stage to score them separately. If retrieval scores are low, I have several levers. First, chunk size — chunks that are too large dilute semantic meaning, too small lose context. I'd run the evaluation across chunk sizes of 400, 800, and 1200 characters and compare retrieval scores. Second, I'd add a reranking step — retrieve the top 10 chunks by embedding similarity, then run a cross-encoder reranker to rescore them and keep only the top 3 for generation. Cross-encoders are slower but significantly more accurate than bi-encoders for relevance scoring. Third, I'd look at query reformulation — if user queries are short and abstract, expanding them before retrieval using an LLM improves embedding match quality. If generation scores are low despite good retrieval, that's a prompt engineering problem — I'd tighten the system prompt constraints and add few-shot examples of good answers."


pip install fpdf2
```

Add to requirements.txt:
```
fpdf2
```

Run it and you get 90 realistic fake German insurance claim PDFs you can put on GitHub safely.

---

## The Professional Project Story

Here is your interview narrative. Memorize the structure, not the exact words:

---

**Opening — what and why:**
"I built an end-to-end document intelligence pipeline for insurance claims processing. The motivation was real — insurance companies like Allianz Austria receive thousands of claim documents in multiple formats and languages. The manual processing is slow, error-prone, and doesn't scale. I wanted to build a system that could automatically ingest, classify, extract structured information, and make it semantically searchable."

**The data challenge:**
"The documents are real-world German insurance claims — emails, multi-page PDFs with scanned attachments, mixed content. This created immediate challenges: scanned pages returning no text, German language requiring multilingual embeddings, and documents ranging from 2 to 100 pages requiring intelligent chunking rather than naive splitting."

**Stage 1 — Ingestion:**
"The first stage handles PDF ingestion with per-document error isolation. A critical design decision was wrapping each document in its own try-except rather than the whole batch — if document 47 fails, documents 48 through 89 still process. I also track which pages failed text extraction, because insurance PDFs commonly have scanned image pages. This metadata flows through every subsequent stage."

**Stage 2 — Extraction:**
"Stage 2 uses GPT-4o to classify each document and extract structured fields — claim number, date, sender, damage type, required actions. I used Pydantic models to validate the LLM output before it enters the pipeline. This was important because without validation, a missing field wouldn't cause a failure until Stage 5 when you're trying to display it — hard to debug and potentially serving wrong answers to users. I also prompt the model to generate an English summary regardless of source language, which simplifies downstream retrieval."

**Stage 3 — Chunking:**
"I implemented fixed-size chunking with overlap and sentence boundary detection. A bug I caught during development: the overlap calculation could land mid-word, producing chunks starting with fragments like 'chenfehler' instead of complete words. I fixed this by searching for the next word boundary after calculating the overlap position. I also handle short documents as single chunks — no point splitting a 200-character email into fragments."

**Stage 4 — Embedding:**
"I chose sentence-transformers over OpenAI embeddings for two reasons: cost and multilingual capability. The paraphrase-multilingual-MiniLM-L12-v2 model embeds German and English in the same vector space, meaning an English query finds German chunks when they're semantically equivalent. ChromaDB stores vectors with rich metadata — claim number, urgency, date, sender — enabling hybrid search: semantic similarity plus metadata filtering."

**Stage 5 — Retrieval:**
"The RAG query stage retrieves top-N chunks and passes them as context to GPT-4o with a strict system prompt — answer only from context, never hallucinate, always cite claim numbers. Every query is logged with the retrieved chunks, distances, and generated answer. This logging is what makes evaluation possible."

**Stage 6 — Evaluation:**
"This is where my background in model evaluation at Allianz gave me an advantage. I built an LLM-as-judge evaluator that scores retrieval and generation separately — a distinction most tutorials skip. Retrieval failure means wrong chunks came back, fix the embedding or chunking. Generation failure means right chunks but bad answer, fix the prompt. The evaluator produces an actionable report sorted by worst-performing queries with specific improvement suggestions per query."

**Infrastructure:**
"The entire pipeline is containerized with Docker. Each stage runs as an independent container sharing data through volume mounts. This means any stage can be rerun independently without reprocessing everything — important when you have 89 documents and stage 6 fails, you don't want to re-embed everything. I then added Celery with Redis as a message broker to convert the sequential pipeline into an async system where each document travels through all stages as an independent chain. This enables processing documents concurrently — at 10,000 documents the difference between sequential and async is hours versus days."

**What I learned and would improve:**
"The most valuable insight was building evaluation before optimizing retrieval. I saw a query scoring 0.739 average distance — poor retrieval — and instead of tuning randomly I had data showing exactly which query failed and why. For improvement I'd add a cross-encoder reranker between retrieval and generation, implement query expansion for abstract queries, and replace ChromaDB with Qdrant for production scale."

---

## On the Next Project

You're right that LangChain and LangGraph appear constantly in job descriptions. The next project should use them deliberately so you can speak to both approaches — "I've built RAG without frameworks to understand the fundamentals, and with LangGraph to understand agent orchestration."

Start a new conversation for that. Tell me the industry you want to target — healthcare, legal, finance, e-commerce — and what kind of problem you want to solve. I'll design a project that specifically hits the LangGraph, agents, and tool-use keywords that job descriptions are asking for right now.

One last thing before you go — update your README with the professional story above, adapted in your own words. And when you push to GitHub, use the synthetic data generator first, verify the pipeline runs end-to-end on synthetic data, then publish. Never commit the real Allianz data even accidentally — add `data/pdfs/` to your `.gitignore` immediately.
```
# .gitignore
.env
data/pdfs/
data/output/
chroma_db/
logs/
__pycache__/
*.pyc
venv/
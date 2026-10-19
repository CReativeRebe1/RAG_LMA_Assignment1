# How the Notebook Works

This file shows how data moves through `rag_baseline_vs_rerank.ipynb`, step by step, and which function does what.

## 1. Preparing the document (done once)

```
data/dpdp_act_2023.pdf   (the Act, 21 pages; downloaded automatically if missing)
   │
   ├─ read_margin_titles()     finds the 44 section titles printed in the page margins
   │                           (skips margin notes that only name other Acts, like "24 of 1997.")
   │
   ├─ read_body_paragraphs()   reads the main text line by line and joins the lines back into paragraphs
   │                           (is_noise() removes page headers, the front-page banner, footer and signature)
   │
   └─ read_schedule_rows()     reads the penalty table on the last page, one row per fine (7 rows)
   │
   ▼
split_into_sections()   →  44 sections + the Schedule, each with its number, title and chapter
   │
   ▼
make_chunks()           →  105 chunks, each at most 200 tokens
                           every chunk starts with "Section N – Title."        saved to results/chunks.jsonl
   │
   ▼
embed()                 →  turns each chunk into a vector (105 vectors of size 384)
FAISS index             →  stores the vectors so we can search them
```

## 2. Answering a question

Both pipelines start with the same search. Only what happens after it is different.

```
question
   │
   ▼
dense_top_k()   encodes the question and finds the 10 closest chunks (the "top 10")
   │
   ├── Pipeline 1 – Baseline:  run_baseline()
   │      take the first 3 of the top 10
   │      → build_prompt() → ask_llm() → Answer 1
   │
   └── Pipeline 2 – Re-ranking:  run_rerank()
          cross_rerank() scores each of the 10 chunks together with the question,
          then sorts them by that score
          take the new first 3
          → build_prompt() (the same function) → ask_llm() → Answer 2
```

- **ask_llm():** sends the prompt to `openai/gpt-oss-120b` on Groq (temperature 0). The API key is read from `.env` by `load_env_file()`. Every answer is saved in `results/llm_cache.json`, so running again gives the same answers without calling the API.
- **What each pipeline returns:** the question, its top 10, its top 3, the prompt, the answer, and how long each step took.

## 3. Testing and measuring

```
32 test questions (Q01–Q20 first, Q21–Q32 added later)
   each has: a category, the question, a phrase from the Act, and the real answer
   │
   ▼
gold_chunk_ids()    finds the correct chunk(s) for each question using its phrase
   │
   ▼
for every question: run both pipelines and note the rank of the correct chunk
   best_rank()  → rank of the correct chunk in each pipeline
   outcome()    → fixed / made worse / both right / both wrong / not found in top 10
                                                         saved to results/rank_shift.csv
   │
   ▼
hit_at(), mrr()     → how often the right chunk is at rank 1, in the top 3, in the top 10, and MRR
                                                         saved to results/metrics.csv
   │
   ▼
show the full top 10 for every question that re-ranking fixed or made worse (show_candidates())
   │
   ▼
get Answer 1 and Answer 2 for all 32 questions          saved to results/answers.json
answers checked by hand (correct / refused / ...)       saved to results/answer_review.csv
count of each label                                     saved to results/verdict_summary.csv
   │
   ▼
profile_latency()   times each step 30 times per question, on the GPU and on the CPU
                                                         saved to results/latency.csv
scaling test        times the cross-encoder on 10, 100 and 1,000 chunks and estimates bigger sizes;
                    times the FAISS search on up to 100,000 vectors
                                                         saved to results/scaling.csv
charts                                                  saved to results/figures/
```

## 4. All functions

| Function | What it does | Notebook section |
|---|---|---|
| `is_noise` | tells if a line is a header, footer or other junk | 2 |
| `read_margin_titles` | gets the 44 section titles from the margins | 2 |
| `read_body_paragraphs` | gets the main text as paragraphs | 2 |
| `read_schedule_rows` | gets the 7 rows of the penalty table | 2 |
| `split_into_sections` | groups paragraphs into the 44 sections + the Schedule | 3 |
| `make_chunks` | cuts sections into chunks of at most 200 tokens | 3 |
| `embed` | turns text into vectors | 4 |
| `dense_top_k` | finds the 10 closest chunks for a question | 5 |
| `cross_rerank` | re-scores and re-sorts those 10 with the cross-encoder | 5 |
| `show_candidates` | shows a table of the 10 chunks with both ranks | 5 |
| `build_prompt` | puts the question and 3 chunks into the prompt (same for both pipelines) | 6 |
| `load_env_file` | reads the API key from `.env` | 6 |
| `ask_llm` | gets an answer from the LLM, or from the saved answers | 6 |
| `run_baseline` / `run_rerank` | runs one full pipeline for a question | 7 |
| `gold_chunk_ids` | finds the correct chunk(s) for a question | 8 |
| `best_rank` / `outcome` | finds the correct chunk's rank and labels the result | 9 |
| `hit_at` / `mrr` | calculates the scores | 9 |
| `profile_latency` | times each step | 11 |
| `median_ms` | times one function several times and returns the middle value | 11 |

## 5. The report

`report/report.pdf` uses the numbers from the `results/*.csv` files and two charts from `results/figures/`
(`rank_shift_slope.png` and `scaling.png`).

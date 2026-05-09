# UChicago MS in Applied Data Science — RAG Chatbot

A conversational AI assistant built for the Master's in Applied Data Science program at the University of Chicago. The system answers questions about courses, admissions, tuition, faculty, career outcomes, and more by retrieving information directly from the program's official website and generating source-cited responses.

---

## What It Does

Most university chatbots either hallucinate answers or just redirect you to a webpage. This system takes a different approach: it scrapes the actual MS-ADS website, stores the content in a vector database, and when you ask a question, it finds the most relevant passages and feeds them to a large language model to generate an accurate, cited answer.

The result is a chatbot that can answer questions like:

- "What are the core courses and what do they cover?"
- "What is the difference between the online and in-person programs?"
- "What career outcomes have past graduates achieved?"
- "How much does the program cost and is financial aid available?"
- "What does the capstone project involve?"

Every answer includes the source URL so that you can verify the information directly.

---

## Architecture

The system follows a standard RAG (Retrieval-Augmented Generation) pipeline:

```
MS-ADS Website
      |
      v
Web Scraper (BeautifulSoup)
      |
      v
Text Chunks (150 words, 30-word overlap)
      |
      v
Embeddings (BAAI/bge-small-en-v1.5)
      |
      v
Vector Store (ChromaDB)

                    User Question
                          |
                          v
                  Retrieve Top-8 Chunks
                          |
                          v
                  GPT-4.1-mini (Azure)
                          |
                          v
                  Grounded Answer + Sources
```

The key principle is that the language model only answers from the retrieved content. It is explicitly instructed not to use prior training knowledge, which prevents hallucination of course names, tuition figures, or deadlines that the model might have seen during training but could be outdated.

---

## Project Structure

```
uchicago-msads-rag/
|
|-- src/
|   |-- scraper.py          Web crawler for the MS-ADS website
|   |-- embedder.py         Embedding model + ChromaDB vector store
|   |-- llm_factory.py      LLM factory (Azure OpenAI or HuggingFace)
|   |-- rag_chain.py        Conversational RAG pipeline
|   |-- evaluator.py        Evaluation (heuristic, BLEU, ROUGE, RAGAS)
|
|-- ui/
|   |-- app.py              Streamlit chatbot interface
|
|-- RAG_Pipeline_Final_v5.ipynb    Full end-to-end Colab notebook
|-- requirements.txt
|-- .env.example
|-- README.md
```

---

## Tech Stack

| Component | Technology | Notes |
|---|---|---|
| Web scraping | requests + BeautifulSoup4 | BFS crawler, 13 seed URLs |
| Embeddings | BAAI/bge-small-en-v1.5 | Free, 384-dimensional, runs on CPU |
| Vector store | ChromaDB | Persisted locally |
| LLM | Azure OpenAI GPT-4.1-mini | Student credits; switchable to HuggingFace |
| Orchestration | LangChain 0.2 | Chain composition, prompt templates, memory |
| UI | Streamlit + ngrok | Chat interface with source citation display |
| Evaluation | Heuristic + RAGAS + BLEU/ROUGE | Multi-method evaluation suite |

---

## Key Features

**Conversational memory.** The system remembers the last 10 turns of conversation. If you ask "What are the admission requirements?" and then follow up with "Do I need work experience?", the second question is automatically rewritten as "Does the MS-ADS program require work experience?" before being searched. This means follow-up questions work naturally without you having to repeat context.

**Source citation.** Every answer includes the URL of the page(s) the information came from. The Streamlit UI also lets you expand a panel to read the raw retrieved text chunks, so you can see exactly what the model was given.

**Responsible AI guardrails.** The system detects off-topic questions and redirects them. It strips PII (email addresses, phone numbers, SSNs) from generated responses. The system prompt explicitly prohibits the model from inventing course names, deadlines, or statistics.

**Curated course index.** The core and elective course names are added as hand-crafted summary chunks in addition to the scraped chunks. This ensures that short queries like "what are the core courses?" retrieve the right content even when the scraped chunks spread course descriptions across many individual paragraphs.

**Switchable LLM backend.** Setting `LLM_PROVIDER = 'huggingface'` in the credentials cell switches the entire pipeline to use Mistral-7B-Instruct for free, with no code changes required.

---

## Evaluation

The system is evaluated using four complementary methods, each measuring something different.

**Heuristic evaluation** runs instantly with no API calls. It scores answers on whether a real answer was given (not a fallback), whether enough context was retrieved, keyword overlap between question and answer, absence of hedging language, and whether source URLs were cited.

**BLEU and ROUGE** measure n-gram overlap between generated answers and reference answers. BLEU measures precision (how much of the generated text matches the reference), while ROUGE measures recall (how much of the reference is covered). These scores tend to be lower than you might expect because GPT-4.1-mini paraphrases rather than copying the reference wording, which is actually the correct behavior.

**LLM-as-Judge** asks the same language model to score each answer on five dimensions: correctness, completeness, relevance, clarity, and grounding. Each dimension is scored 0-10 for a total of 0-50. This method captures semantic quality that BLEU/ROUGE cannot measure since it understands paraphrasing.

**RAGAS** provides RAG-specific metrics: faithfulness (is the answer supported by the retrieved context?), answer relevancy, context precision (are the retrieved chunks relevant?), and context recall (does the retrieved context cover the ground truth?). This is the most targeted evaluation for a RAG system because it evaluates both the retrieval and the generation together.

---

## Running the Project

The entire pipeline runs in a single Google Colab notebook. Open `RAG_Pipeline_Final_v5.ipynb` in Colab and run the cells in order.

### Credentials

You will need to add three secrets in Colab's Secrets panel (the key icon in the left sidebar). Toggle "Notebook access" on for each one.

| Secret name | Value |
|---|---|
| AZURE_OPENAI_KEY | Your Azure OpenAI API key |
| AZURE_OPENAI_ENDPOINT | https://YOUR-RESOURCE.openai.azure.com/ |
| AZURE_OPENAI_DEPLOYMENT | Your deployment name, e.g. gpt-4o-mini |

To use HuggingFace instead (no API cost), change `LLM_PROVIDER = 'huggingface'` in Cell 2. The embeddings always use HuggingFace and are always free.

### First-time setup

Run Cell 1 (installs packages), then restart the runtime (Runtime > Restart session), then run from Cell 2 onwards. The runtime restart is required because pinning numpy to avoid a binary incompatibility with ChromaDB requires a clean Python process.

After the first run, the scraped data and vector store are saved to `data/`. If you reconnect to Colab later, skip Cell 9 (scrape) and Cell 11 (embed) if the data files still exist.

### Installation

```bash
pip install -r requirements.txt
```

---

## Design Decisions

**Why 150-word chunks with 30-word overlap?** Larger chunks (the original scraper used 500 words) dilute the embedding because one 500-word chunk might contain course descriptions, Coursera recommendations, and admissions information all mixed together. The embedding of that chunk is an average of all those topics, so it matches poorly against specific queries. At 150 words, each chunk covers roughly one topic, giving the embedding model a focused signal to work with.

**Why curated course chunks?** The course-progressions page presents course names as headings followed by multi-paragraph descriptions. When chunked at 150 words, each chunk contains the description of one course but not the full list. A query like "what are the core courses?" needs to match against a chunk that contains all six course names together. Rather than engineering around this with query expansion or larger chunks, the cleanest solution is to add three hand-crafted summary chunks that list all courses in a clean format.

**Why LangChain 0.2 and not a newer version?** Newer versions of LangChain (0.3+) changed the import paths for several core classes and introduced API changes that conflict with the chromadb and openai versions that are stable in the Google Colab environment as of early 2025. Pinning to 0.2.16 ensures the notebook runs without version conflicts.

---

## Limitations

- The knowledge base is a snapshot of the website from the time of scraping. Course offerings, tuition figures, and application deadlines change each quarter, so the chatbot may give outdated answers for time-sensitive information. Re-running Cell produces a fresh scrape.

- JavaScript-rendered content is not captured. Some parts of the MS-ADS website load content dynamically, which BeautifulSoup cannot access without a headless browser. This primarily affects any interactive elements on the site.

- BLEU and ROUGE scores are intentionally lower than the LLM-as-Judge scores. This is not a problem with the system. It reflects the fact that the model generates correct, well-phrased answers that use different wording than the short reference answers in the test set.

---

## About

Program website: https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/

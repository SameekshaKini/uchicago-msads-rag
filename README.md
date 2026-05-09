# RAG-Based AI Assistant — UChicago MS in Applied Data Science

**GEN AI Principles | Course Project 1**

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about the
[MS in Applied Data Science](https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/)
program at the University of Chicago.

## Project Structure

```
├── src/
│   ├── scraper.py        # Web crawler for the MS-ADS website
│   ├── embedder.py       # Embeddings + ChromaDB vector store
│   ├── llm_factory.py    # LLM factory (Azure OpenAI / HuggingFace)
│   ├── rag_chain.py      # Conversational RAG pipeline
│   └── evaluator.py      # Heuristic + RAGAS evaluation
├── ui/
│   └── app.py            # Streamlit chatbot UI
├── RAG_Pipeline_Final_v5.ipynb   # Full end-to-end notebook (Colab)
└── README.md
```

## How to Run

Open `RAG_Pipeline_Final_v5.ipynb` in Google Colab and run cells top to bottom.

### Requirements
- Azure OpenAI account (student credits) with a `gpt-4o-mini` deployment, or
- HuggingFace account (free) for open-source models

### Colab Secrets needed
| Secret | Value |
|---|---|
| `AZURE_OPENAI_KEY` | Your Azure API key |
| `AZURE_OPENAI_ENDPOINT` | `https://YOUR-RESOURCE.openai.azure.com/` |
| `AZURE_OPENAI_DEPLOYMENT` | e.g. `gpt-4o-mini` |

## Tech Stack
| Component | Technology |
|---|---|
| Web scraping | BeautifulSoup + requests |
| Embeddings | BAAI/bge-small-en-v1.5 (free, HuggingFace) |
| Vector store | ChromaDB |
| LLM | Azure OpenAI GPT-4o-mini |
| Orchestration | LangChain 0.2 |
| UI | Streamlit + ngrok |
| Evaluation | Heuristic + RAGAS |

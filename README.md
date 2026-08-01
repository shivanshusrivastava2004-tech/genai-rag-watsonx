# GenAI Document Q&A Assistant (RAG + IBM watsonx)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![IBM watsonx](https://img.shields.io/badge/IBM-watsonx-054ADA.svg)](https://www.ibm.com/watsonx)
[![LangChain](https://img.shields.io/badge/LangChain-Framework-1C3C3C.svg)](https://www.langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Retrieval-Augmented Generation (RAG) system that connects an **IBM watsonx LLM** to a vector database, enabling accurate natural-language question answering over custom document sets. Built as the applied capstone for the IBM "Gen AI Using watsonx" certification (scored 98/100).

---

## 🎯 Overview

Traditional LLMs hallucinate when asked about content outside their training data. This project solves that by grounding responses in your own documents through a full RAG pipeline — chunking, embedding, semantic retrieval, and prompt-engineered generation.

## 🏗️ Architecture

```
┌─────────────┐     ┌───────────────┐     ┌──────────────────┐     ┌─────────────┐
│  Documents  │ --> │  Chunking &   │ --> │  Vector Database   │ --> │   IBM       │
│  (PDF/TXT)  │     │  Embedding    │     │  (Semantic Search)  │     │  watsonx LLM │
└─────────────┘     └───────────────┘     └──────────────────┘     └─────────────┘
                                                                            │
                                                                            ▼
                                                                  ┌───────────────────┐
                                                                  │  Grounded Answer   │
                                                                  │  (Reduced          │
                                                                  │  Hallucination)    │
                                                                  └───────────────────┘
```

## ✨ Features

- **Document Ingestion** — Parses and chunks documents for optimal retrieval granularity
- **Semantic Embeddings** — Converts chunks into vector representations for similarity search
- **Vector Retrieval** — Fetches the most relevant context for a given query
- **LangChain Orchestration** — Coordinates the retrieval → generation pipeline
- **Prompt Engineering** — Few-shot examples, context windowing, and guardrail instructions to boost accuracy and reduce hallucination
- **IBM watsonx Integration** — Uses enterprise-grade foundation models for generation

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| LLM | IBM watsonx |
| Orchestration | LangChain |
| Vector Store | Chroma / FAISS |
| Language | Python 3.10+ |

## 📦 Installation

```bash
git clone https://github.com/<your-username>/genai-rag-watsonx.git
cd genai-rag-watsonx
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## ⚙️ Configuration

Create a `.env` file in the project root:

```env
WATSONX_API_KEY=your_api_key_here
WATSONX_PROJECT_ID=your_project_id_here
WATSONX_URL=https://us-south.ml.cloud.ibm.com
```

> ⚠️ Never commit your `.env` file. It's already excluded via `.gitignore`.

## 🚀 Usage

```bash
python src/ingest.py --docs ./data/
python src/query.py --question "What are the key findings in the report?"
```

## 📁 Project Structure

```
genai-rag-watsonx/
├── src/
│   ├── ingest.py         # Document loading & chunking
│   ├── embed.py          # Embedding generation
│   ├── retriever.py      # Vector search logic
│   ├── query.py          # Main Q&A entrypoint
│   └── prompts.py        # Prompt templates & guardrails
├── docs/
│   └── architecture.md
├── assets/
│   └── demo.gif
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## 📊 Results

- Reduced hallucination rate through grounded retrieval and prompt guardrails
- Modular pipeline supports swapping vector DBs or LLM backends with minimal changes

## 🎓 Background

Developed as the applied capstone project for the **IBM "Gen AI Using watsonx" certification** (Issued June 2025, Score: 98/100).

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## 👤 Author

**Shivanshu Srivastava**
[LinkedIn](https://linkedin.com/in/shivanshu-srivastava) · shivanshu.srivastava2004@gmail.com

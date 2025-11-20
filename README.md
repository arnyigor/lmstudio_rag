# Local Knowledge Base Manager (RAG)

A lightweight, command‑line tool for building a **retrieval‑augmented generation** (RAG) knowledge base on your local machine.  
It uses:

* **LM Studio** – a fast local LLM server that exposes an OpenAI‑compatible API.
* **ChromaDB** – a vector store with HNSW indexing, ideal for similarity search.

> The script works entirely offline; no cloud calls are required.

---

## 📦 Prerequisites

| Component | Minimum version |
|-----------|-----------------|
| Python     | 3.8+            |
| LM Studio  | ≥ 0.1 (or any local server that implements the OpenAI API) |
| ChromaDB   | `chromadb==latest` |

### Install dependencies

```bash
# Optional: create a virtual environment first
python -m venv .venv
source .venv/bin/activate      # Windows: .\.venv\Scripts\activate

pip install -r requirements.txt
```

**requirements.txt**

```text
chromadb>=0.4
openai>=1.3
pypdf>=3.17
python-docx>=1.1
tqdm>=4.66
colorama>=0.4
```

---

## ⚙️ Configuration

The script reads a few environment variables.  
If you leave them unset, the defaults in `save_to_chroma.py` will be used.

| Variable | Purpose | Default |
|----------|---------|---------|
| `CHROMA_DB_PATH` | Path where ChromaDB stores its data | `G:\AIModels\chroma_db\chroma_db` |
| `DEFAULT_DOCS_DIR` | Folder that is used when the user presses *Enter* on the file‑path prompt | `G:\Android\ChromaDbDocuments` |
| `LM_STUDIO_URL` | Base URL of the local LM Studio server | `http://localhost:1234/v1` |
| `LM_STUDIO_API_KEY` | API key accepted by LM Studio (usually a dummy value) | `lm-studio` |
| `CHROMA_COLLECTION` | Name of the Chroma collection to use | `my_knowledge_base` |

Example:

```bash
export CHROMA_DB_PATH="/home/user/chroma_db"
export DEFAULT_DOCS_DIR="/home/user/docs"
export LM_STUDIO_URL="http://localhost:1234/v1"
export LM_STUDIO_API_KEY="lm-studio"
```

> **Tip:** If you use Windows PowerShell, prepend `$env:` instead of `export`.

---

## 🚀 Running the script

```bash
python save_to_chroma.py
```

The console shows a menu:

```
=== Менеджер Локальной Базы Знаний (RAG) ===
1. Индексировать папку (Добавить файлы)
2. Удалить текущую коллекцию (Очистить базу)
3. Информация о базе
4. Выход
Выберите действие (1-4):
```

### 1️⃣ Index a folder

*Enter the path to a directory containing `.txt`, `.md`, `.pdf` or `.docx` files.*  
If you just press **Enter**, `DEFAULT_DOCS_DIR` is used.

The script:

1. Scans all files in the folder.
2. Reads each file, handling the four supported formats.
3. Splits the text into chunks (`CHUNK_SIZE=1000`, `CHUNK_OVERLAP=200`).
4. For every chunk:
   * Generates an embedding via LM Studio (`text‑embedding-nomic-embed-text-v1.5@q8_0` by default).
   * Stores it in Chroma along with the chunk text and metadata (`source`, `position`).

> **Note:** The chunk ID is a MD5 hash of `<filename><offset>` – guarantees uniqueness.

### 2️⃣ Delete current collection

Wipes all documents from the active Chroma collection and recreates an empty one.

### 3️⃣ Database info

Shows:

* Path to the database.
* Collection name.
* Total number of vectors.
* A sample of the first five metadata entries (useful for debugging).

### 4️⃣ Exit

Closes the program.

---

## 📚 Behind‑the‑scenes

| Component | What it does |
|-----------|--------------|
| **LM Studio** | Provides a local OpenAI‑compatible endpoint (`/v1/embeddings`). The script uses `openai.OpenAI(...)` to call it. |
| **ChromaDB** | Stores vectors, documents and metadata in an HNSW index. All operations are performed through the Python client. |
| **Chunking** | Splits long text into overlapping chunks so that retrieval is fine‑grained. |
| **Vector generation** | `text-embedding-nomic-embed-text-v1.5@q8_0` (or any model you loaded in LM Studio) – returns a 768‑dimensional float32 vector. |

> If LM Studio reports *“No models loaded”*, load one via the UI or CLI:
> ```bash
> lms load text-embedding-nomic-embed-text-v1.5@q8_0
> ```

---

## 🔧 Extending / Customising

### Changing the chunk size

Edit `CHUNK_SIZE` and `CHUNK_OVERLAP` near the top of `save_to_chroma.py`.

```python
# Chunking (default: 1000 chars, 200 overlap)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
```

### Using a different embedding model

Set `embedding_model` when creating `VectorDBManager`:

```python
db = VectorDBManager(CHROMA_PATH,
                     collection_name="my_knowledge_base",
                     embedding_model="openai/embedding-ada-002")
```

> Ensure the chosen model is loaded in LM Studio.

### Programmatic usage

You can import the classes and use them directly from another script:

```python
from save_to_chroma import VectorDBManager, FileProcessor

db = VectorDBManager(Path("/path/to/chroma_db"), "my_collection",
                     embedding_model="text-embedding-nomic-embed-text-v1.5@q8_0")

processor = FileProcessor()
text = processor.read_file("some.pdf")
chunks = processor.chunk_text(text, "some.pdf")

db.add_documents(chunks)
```

---

## 📊 Querying the collection

While the interactive menu doesn’t expose a query UI, you can retrieve data in code:

```python
# 1️⃣ By ID
res = db.collection.get(ids=["654c800a7508ec118a5ee9eb66f4a608"])
print(res["documents"][0])

# 2️⃣ By text similarity
query_res = db.collection.query(
    query_texts=["Игорь"],
    n_results=3
)
print(query_res["documents"])

# 3️⃣ By metadata filter
res = db.collection.get(where={"source": "about_me.md"}, limit=10)
print(res["metadatas"])
```

---

## 🛠️ Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `No models loaded` error | LM Studio has no embedding model loaded | Load one via UI or CLI (`lms load …`) |
| Embedding generation hangs / slow | Model is large or GPU memory limited | Switch to a lighter model, or run on CPU |
| Empty collection after indexing | File path wrong / file unsupported | Verify the folder and file types; check console logs |
| Duplicate IDs / collisions | Same file processed twice | Ensure unique filenames or change chunk ID logic |

---

## 🎉 What you get

* A **self‑contained knowledge base** that can be queried locally.
* Fast similarity search thanks to Chroma’s HNSW index.
* Zero external dependencies (no cloud, no API keys).
* Ready‑to‑use example for building a RAG pipeline:
   * Generate embeddings → store in Chroma → query by text → feed results into an LLM.

Happy indexing! 🚀
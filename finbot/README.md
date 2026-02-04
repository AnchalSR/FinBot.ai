# FinBot - Financial Advisor Chatbot 💰

A production-ready, end-to-end AI Financial Chatbot powered by Retrieval Augmented Generation (RAG) and Large Language Models (LLMs).

## 🌟 Features

- **RAG-Powered**: Uses FAISS vector database for efficient document retrieval
- **Multi-Model Support**: Works with HuggingFace models or OpenAI API
- **Production-Ready**: Complete backend, frontend, and deployment configs
- **Modular Design**: Clean separation of concerns with type hints and documentation
- **Document Management**: Upload, process, and manage financial documents
- **Source References**: Shows document sources with relevance scores
- **Fine-Tuning Support**: Optional LoRA-based model fine-tuning
- **Streamlit UI**: Beautiful, user-friendly chat interface
- **FastAPI Backend**: RESTful API with comprehensive endpoints
- **Comprehensive Logging**: Production-grade logging throughout

## 📊 Architecture

```
FinBot Architecture
├── Frontend (Streamlit)
│   └── Web UI for chat and document management
├── Backend (FastAPI)
│   ├── REST API endpoints
│   └── RAG Pipeline
├── Core
│   ├── FAISS Vector Database
│   ├── SentenceTransformers
│   └── LLM Integration (HuggingFace/OpenAI)
└── Data Processing
    ├── PDF/TXT Loading
    ├── Text Cleaning
    └── Document Chunking
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip/conda
- (Optional) GPU for faster inference (NVIDIA CUDA)
- (Optional) OpenAI API key
- (Optional) HuggingFace API token

### Installation

1. **Clone the repository**
   ```bash
   cd c:\Users\hp\Desktop\Finbot.ai\finbot
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create environment file**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Prepare documents**
   - Create `data/documents/` folder
   - Add PDF or TXT files with financial information
   - Example: `data/documents/finance_guide.pdf`

### Running the Application

#### Option 1: Run Backend + Frontend (Full Stack)

**Terminal 1: Start Backend API**
```bash
python -m backend.api
```

The API will start on `http://localhost:8000`

Check health: `http://localhost:8000/health`
API docs: `http://localhost:8000/docs`

**Terminal 2: Start Streamlit Frontend**
```bash
streamlit run frontend/app.py
```

The web UI will open on `http://localhost:8501`

#### Option 2: Use Backend Only (API)

```bash
python -m backend.api
```

Then interact using curl or any HTTP client:

```bash
# Health check
curl http://localhost:8000/health

# Chat endpoint
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is compound interest?",
    "top_k": 5,
    "include_sources": true
  }'

# Upload documents
curl -X POST http://localhost:8000/upload \
  -F "files=@data/documents/finance_guide.pdf"

# Get status
curl http://localhost:8000/status

# List documents
curl http://localhost:8000/documents
```

#### Option 3: Use Frontend Only (Requires Running Backend)

```bash
streamlit run frontend/app.py
```

## 📁 Project Structure

```
finbot/
├── data/
│   └── documents/          # Input financial documents (PDF/TXT)
├── embeddings/             # FAISS vector index storage
├── backend/
│   ├── api.py              # FastAPI application
│   └── rag.py              # RAG pipeline implementation
├── frontend/
│   └── app.py              # Streamlit web interface
├── training/
│   └── finetune.py         # Model fine-tuning (optional)
├── utils/
│   └── preprocess.py       # Document processing utilities
├── config/
│   └── settings.py         # Configuration management
├── logs/                   # Application logs
├── .env.example            # Environment variables template
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── deploy.md              # Deployment guide
```

## 🔧 Configuration

### Environment Variables (.env)

Key configurations:

```bash
# LLM Choice
USE_OPENAI=false                    # Set to true for OpenAI
OPENAI_API_KEY=sk-...               # Your OpenAI key
HF_MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.1  # HuggingFace model

# Embedding
EMBEDDING_MODEL=all-MiniLM-L6-v2    # Fast, accurate embeddings

# RAG Parameters
CHUNK_SIZE=500                      # Document chunk size
CHUNK_OVERLAP=50                    # Overlap between chunks
TOP_K_DOCUMENTS=5                   # Documents to retrieve

# Performance
DEVICE=cpu                          # or cuda for GPU
MAX_TOKENS=512                      # Max response length
TEMPERATURE=0.7                     # Response creativity
```

See [.env.example](.env.example) for all options.

## 💬 API Endpoints

### Core Endpoints

#### 1. Chat Endpoint
```
POST /chat
Content-Type: application/json

Request:
{
  "query": "What is compound interest?",
  "top_k": 5,
  "include_sources": true
}

Response:
{
  "answer": "Compound interest is...",
  "sources": [
    {
      "source": "finance_guide.pdf",
      "similarity": 0.95,
      "chunk_id": 0
    }
  ],
  "query": "What is compound interest?",
  "timestamp": "2024-02-03T10:30:00"
}
```

#### 2. Health Check
```
GET /health

Response:
{
  "status": "healthy",
  "timestamp": "2024-02-03T10:30:00",
  "documents_count": 42,
  "embedding_model": "all-MiniLM-L6-v2"
}
```

#### 3. Status Endpoint
```
GET /status

Response:
{
  "status": "ready",
  "rag_initialized": true,
  "documents_count": 42,
  "embedding_dimension": 384,
  "device": "cpu",
  "embedding_model": "all-MiniLM-L6-v2"
}
```

#### 4. Upload Documents
```
POST /upload
Content-Type: multipart/form-data

Files: document1.pdf, document2.txt

Response:
{
  "status": "success",
  "uploaded_files": ["document1.pdf", "document2.txt"],
  "message": "Files uploaded and will be processed in background"
}
```

#### 5. List Documents
```
GET /documents

Response:
{
  "total_documents": 42,
  "sources": {
    "finance_guide.pdf": 10,
    "investment_basics.txt": 8
  }
}
```

#### 6. Delete Document Source
```
DELETE /documents/{source}

Example: DELETE /documents/finance_guide.pdf

Response:
{
  "status": "success",
  "removed_documents": 10,
  "remaining_documents": 32
}
```

## 🎓 Usage Examples

### Python API Usage

```python
from backend.rag import create_rag_pipeline

# Create pipeline
rag = create_rag_pipeline()

# Load documents
rag.load_documents_from_folder("data/documents")

# Retrieve documents
results = rag.retrieve_documents("What is stock diversification?", top_k=5)

# Build context
context, docs = rag.build_context("What is stock diversification?")

# Get statistics
stats = rag.get_stats()
print(f"Total documents: {stats['total_documents']}")
```

### Using from Command Line

```bash
# Start backend
python -m backend.api

# In another terminal, upload documents
curl -X POST http://localhost:8000/upload \
  -F "files=@my_financial_guide.pdf"

# Ask a question
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I start investing?"}'
```

## 🤖 LLM Configuration

### Using HuggingFace (Default)

```bash
USE_OPENAI=false
HF_MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.1
HF_API_TOKEN=hf_your_token_here
DEVICE=cpu  # or cuda for GPU
```

**Recommended Models:**
- `mistralai/Mistral-7B-Instruct-v0.1` - Fast, accurate
- `meta-llama/Llama-2-7b-chat` - High quality
- `google/flan-t5-large` - Smaller, faster

### Using OpenAI

```bash
USE_OPENAI=true
OPENAI_API_KEY=sk_your_key_here
OPENAI_MODEL=gpt-3.5-turbo
```

## 📚 Document Processing

### Supported Formats
- **PDF**: Automatically extracted using PyPDF2
- **TXT**: Plain text files with UTF-8 encoding

### Processing Pipeline

1. **Loading**: Extract text from documents
2. **Cleaning**: Remove noise, normalize whitespace
3. **Chunking**: Split into overlapping chunks (500 chars, 50 overlap)
4. **Embedding**: Convert to vectors using SentenceTransformers
5. **Indexing**: Store in FAISS for fast retrieval

### Example Document Structure

```
data/documents/
├── financial_basics.pdf
├── investment_guide.txt
└── risk_management.pdf
```

## 🔬 Fine-Tuning (Optional)

Improve model performance on financial questions:

### Prepare Training Data

Create `data/financial_qa.json`:

```json
[
  {
    "question": "What is compound interest?",
    "answer": "Compound interest is interest calculated on the initial amount and accumulated interest.",
    "context": "Financial concepts"
  },
  {
    "question": "How do stocks work?",
    "answer": "Stocks represent ownership in a company.",
    "context": "Investment basics"
  }
]
```

### Run Fine-Tuning

```python
from training.finetune import FineTuner

fine_tuner = FineTuner(
    model_name="mistralai/Mistral-7B-Instruct-v0.1",
    device="cuda"  # Use GPU for faster training
)

fine_tuner.load_model()
train_loader, val_loader = fine_tuner.prepare_data("data/financial_qa.json")

history = fine_tuner.fine_tune(
    train_loader,
    val_loader,
    epochs=3,
    learning_rate=2e-4,
    output_dir="checkpoints"
)

fine_tuner.save_model("models/finbot-finetuned")
```

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t finbot:latest .
```

### Run Container

```bash
docker run -p 8000:8000 -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/embeddings:/app/embeddings \
  --env-file .env \
  finbot:latest
```

## 🚢 Cloud Deployment

### Streamlit Cloud

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Create new app, select GitHub repo and `frontend/app.py`
4. Set secrets (OPENAI_API_KEY, etc.)

### HuggingFace Spaces

1. Create new Space on HuggingFace
2. Upload your code
3. Create `README.md` with `---\ntitle: FinBot\n---`
4. Add environment variables in Space settings

### AWS Lambda + API Gateway

```bash
# Package backend for Lambda
pip install zappa
zappa init
zappa deploy production
```

### Google Cloud Run

```bash
gcloud run deploy finbot-api --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

See [deploy.md](deploy.md) for detailed deployment instructions.

## 📊 Performance Optimization

### CPU Optimization

```bash
DEVICE=cpu
EMBEDDING_MODEL=all-MiniLM-L6-v2  # Smaller, faster model
CHUNK_SIZE=250                     # Smaller chunks
TOP_K_DOCUMENTS=3                  # Fewer documents
```

### GPU Optimization

```bash
DEVICE=cuda
EMBEDDING_MODEL=all-mpnet-base-v2  # Larger, more accurate
CHUNK_SIZE=1000                    # Larger chunks
TOP_K_DOCUMENTS=10                 # More documents
```

## 🐛 Troubleshooting

### API Connection Issues

**Error:** "Cannot connect to API at http://localhost:8000"

**Solution:**
```bash
# Make sure backend is running in another terminal
python -m backend.api

# Check if port 8000 is available
netstat -ano | findstr :8000  # Windows
lsof -i :8000                 # macOS/Linux
```

### Out of Memory (OOM)

**Solution:**
```bash
DEVICE=cpu                     # Use CPU instead
CHUNK_SIZE=250                 # Smaller chunks
BATCH_SIZE=8                   # Smaller batches
```

### FAISS Index Issues

**Solution:**
```bash
# Rebuild index
python -c "from backend.rag import create_rag_pipeline; \
  rag = create_rag_pipeline(); \
  rag.clear_index(); \
  rag.load_documents_from_folder('data/documents')"
```

### Model Loading Errors

**Error:** "Model not found on HuggingFace"

**Solution:**
```bash
# Check model name
HF_MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.1

# Login to HuggingFace if using gated models
huggingface-cli login
```

## 📈 Monitoring & Logging

### Log Files

```bash
logs/finbot.log        # Application logs
logs/api.log           # API logs
logs/rag.log           # RAG pipeline logs
```

### Monitor Performance

```python
from backend.rag import create_rag_pipeline

rag = create_rag_pipeline()
stats = rag.get_stats()

print(f"Documents: {stats['total_documents']}")
print(f"Embedding dim: {stats['embedding_dimension']}")
print(f"Device: {stats['device']}")
```

## 🧪 Testing

### Unit Tests

```bash
pytest tests/
```

### Integration Tests

```bash
# Test API endpoints
python tests/test_api.py

# Test RAG pipeline
python tests/test_rag.py
```

### Load Testing

```bash
# Using locust
pip install locust
locust -f tests/loadtest.py --host=http://localhost:8000
```

## 📝 Examples

### Example 1: Basic Chat

```python
import requests

response = requests.post(
    "http://localhost:8000/chat",
    json={
        "query": "What is diversification?",
        "top_k": 3
    }
)

print(response.json())
```

### Example 2: Upload and Chat

```python
# Upload documents
files = [('files', open('finance.pdf', 'rb'))]
requests.post("http://localhost:8000/upload", files=files)

# Ask question about uploaded document
response = requests.post(
    "http://localhost:8000/chat",
    json={"query": "Based on the documents, what are the main investment strategies?"}
)
```

### Example 3: Stream API

```python
import requests
from sse_starlette.responses import EventSourceResponse

def stream_chat():
    response = requests.post(
        "http://localhost:8000/chat",
        json={"query": "What is financial planning?"},
        stream=True
    )
    
    for line in response.iter_lines():
        if line:
            print(line.decode('utf-8'))

stream_chat()
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - See LICENSE file for details

## ⭐ Acknowledgments

- [LangChain](https://langchain.com/)
- [HuggingFace](https://huggingface.co/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Streamlit](https://streamlit.io/)
- [FastAPI](https://fastapi.tiangolo.com/)

## 📞 Support

- Documentation: See docs/ folder
- Issues: GitHub Issues
- Discussions: GitHub Discussions

## 🎯 Roadmap

- [ ] Multi-language support
- [ ] Real-time streaming responses
- [ ] Advanced caching
- [ ] User authentication
- [ ] Analytics dashboard
- [ ] Custom model training UI
- [ ] WebSocket support

---

**FinBot v1.0.0** - Built with ❤️ for financial education

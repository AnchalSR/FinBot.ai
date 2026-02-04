# FinBot Quick Reference Guide

## Installation (5 minutes)

```bash
# 1. Navigate to project
cd finbot

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate                    # Linux/macOS
# or: venv\Scripts\activate                # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 5. Create directories
mkdir -p data/documents embeddings logs
```

## Running (Continuous)

### Terminal 1 - Backend API
```bash
source venv/bin/activate
python -m backend.api
```
**Access at:** http://localhost:8000

### Terminal 2 - Frontend
```bash
source venv/bin/activate
streamlit run frontend/app.py
```
**Access at:** http://localhost:8501

### Or use Docker
```bash
docker-compose up -d
```

## Adding Documents

```bash
# Copy your PDF/TXT files to:
cp my_document.pdf data/documents/
cp my_guide.txt data/documents/

# Backend will automatically load them on startup
```

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Check API status |
| `/status` | GET | System statistics |
| `/chat` | POST | Ask questions |
| `/upload` | POST | Upload documents |
| `/documents` | GET | List documents |
| `/documents/{source}` | DELETE | Remove source |
| `/docs` | GET | Swagger API docs |

## Configuration Options

| Setting | Default | Options |
|---------|---------|---------|
| `USE_OPENAI` | false | true/false |
| `DEVICE` | cpu | cpu, cuda |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | any SBERT model |
| `TOP_K_DOCUMENTS` | 5 | 1-20 |
| `CHUNK_SIZE` | 500 | 250-2000 |
| `TEMPERATURE` | 0.7 | 0.0-1.0 |

## Common Commands

```bash
# Check API health
curl http://localhost:8000/health

# Ask a question
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is diversification?"}'

# Upload document
curl -X POST http://localhost:8000/upload \
  -F "files=@document.pdf"

# Get status
curl http://localhost:8000/status

# View logs
tail -f logs/finbot.log
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Module not found | `pip install -r requirements.txt` |
| Can't connect to API | Ensure `python -m backend.api` is running |
| Out of memory | Set `DEVICE=cpu`, reduce `CHUNK_SIZE` |
| No documents | Add files to `data/documents/` |
| Slow inference | Use GPU with `DEVICE=cuda` |
| Port already in use | Change `API_PORT` or `STREAMLIT_PORT` in .env |

## File Locations

| File | Purpose |
|------|---------|
| `config/settings.py` | Configuration management |
| `backend/api.py` | FastAPI server |
| `backend/rag.py` | RAG pipeline |
| `frontend/app.py` | Streamlit UI |
| `utils/preprocess.py` | Document processing |
| `training/finetune.py` | Model fine-tuning |
| `.env` | Environment variables (create from .env.example) |
| `data/documents/` | Input documents go here |
| `embeddings/` | FAISS index stored here |
| `logs/` | Application logs |

## Deployment

### Streamlit Cloud
```bash
git push  # Push to GitHub
# Go to share.streamlit.io, create new app
```

### Docker
```bash
docker-compose up -d      # Start
docker-compose down       # Stop
docker-compose logs -f    # View logs
```

### AWS/GCP
See `deploy.md` for detailed instructions

## Performance Tips

**Faster (CPU):**
```
USE_OPENAI=false
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=250
TOP_K_DOCUMENTS=3
DEVICE=cpu
```

**Better Quality:**
```
USE_OPENAI=true
OPENAI_API_KEY=sk_...
DEVICE=cuda
CHUNK_SIZE=1000
TOP_K_DOCUMENTS=10
```

## API Keys (Keep Secret!)

```bash
# Get OpenAI API key from: https://platform.openai.com/api-keys
# Get HuggingFace token from: https://huggingface.co/settings/tokens

# Store in .env (never commit to git):
OPENAI_API_KEY=sk_xxxxxxx
HF_API_TOKEN=hf_xxxxxxx
```

## Directory Structure

```
finbot/
├── backend/          # FastAPI server & RAG
├── frontend/         # Streamlit UI
├── config/          # Settings
├── utils/           # Helper functions
├── training/        # Fine-tuning
├── data/documents/  # Your PDF/TXT files (create this)
├── embeddings/      # Vector index (auto-created)
├── logs/            # App logs (auto-created)
├── .env             # Your config (create from .env.example)
└── requirements.txt # Dependencies
```

## Key Features

✅ RAG with FAISS vector search
✅ Support for PDF & TXT files
✅ HuggingFace & OpenAI integration
✅ Document chunking with overlap
✅ Source attribution
✅ REST API
✅ Web UI
✅ Docker support
✅ Production logging
✅ Configurable settings

## Next Steps

1. **Setup** → Run installation commands above
2. **Configure** → Edit .env file
3. **Add Documents** → Put PDFs/TXTs in `data/documents/`
4. **Run** → Start backend and frontend
5. **Test** → Ask questions in UI
6. **Deploy** → Use docker-compose or cloud platforms

## Support

- 📖 **README.md** - Full documentation
- 🚀 **deploy.md** - Deployment guide
- ⚙️ **SETUP_GUIDE.md** - Configuration details
- ✅ **MANUAL_TASKS.md** - Complete checklist

---

**FinBot is ready to use! Start with the installation steps above.** 🚀

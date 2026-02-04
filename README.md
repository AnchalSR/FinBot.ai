# Finbot.ai

A financial chatbot powered by open-source language models and retrieval-augmented generation (RAG). Get intelligent financial insights without API costs using Mistral 7B or OpenAI models.

## Features

- 💬 **AI-Powered Financial Chat**: Ask questions about finance, investments, and market trends
- 🚀 **Local & Cloud Options**: Run locally with Mistral 7B or use OpenAI API
- 📚 **RAG Pipeline**: Retrieval-augmented generation for context-aware responses
- 🔐 **Privacy-First**: Process documents locally with configurable models
- ⚡ **CPU/GPU Support**: Run efficiently on CPU or accelerate with CUDA
- 📊 **Vector Search**: Fast semantic search with embedding models

## Tech Stack

- **LLM**: Mistral 7B (Hugging Face) or OpenAI GPT
- **Embeddings**: all-MiniLM-L6-v2
- **Framework**: Python with PyTorch/Transformers
- **Vector Store**: Configurable (Chroma, FAISS, etc.)
- **Deployment**: Docker, cloud-ready

## Prerequisites

- Python 3.8+
- 8GB RAM (minimum for CPU inference), 4GB VRAM (for GPU)
- Git
- HuggingFace account (free) at https://huggingface.co

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/finbot.ai.git
cd finbot.ai
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
# Copy the example file
cp .env.example .env

# Edit .env with your settings
# nano .env  (or use your editor)
```

**Required settings in `.env`:**
- `HF_TOKEN`: Get from https://huggingface.co/settings/tokens (create new token)
- `USE_OPENAI`: Set to `false` for local Mistral, `true` for OpenAI
- `DEVICE`: Set to `cpu` or `cuda` based on your hardware

### 5. Download Models (First Run)

```bash
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.1'); \
AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.1')"
```

## How to Run Locally

### Start the Application

```bash
python main.py
```

Or if using a Streamlit frontend:

```bash
streamlit run app.py
```

### Query the Chatbot

```python
from finbot import FinancialBot

bot = FinancialBot()
response = bot.query("What are the best investment strategies for 2024?")
print(response)
```

## Configuration Options

Edit `.env` to customize:

| Variable | Description | Default |
|----------|-------------|---------|
| `USE_OPENAI` | Use OpenAI instead of local model | `false` |
| `DEVICE` | Inference device | `cpu` |
| `HF_MODEL_NAME` | HuggingFace model | `mistralai/Mistral-7B-Instruct-v0.1` |
| `EMBEDDING_MODEL` | Embedding model for search | `all-MiniLM-L6-v2` |
| `CHUNK_SIZE` | Document chunk size | `500` |
| `TOP_K_DOCUMENTS` | Retrieved context documents | `5` |
| `HF_TOKEN` | HuggingFace API token | Required |

## Deployment

### Docker Deployment

```bash
docker build -t finbot.ai .
docker run -e HF_TOKEN=your_token finbot.ai
```

### Cloud Deployment Options

#### Hugging Face Spaces
1. Fork to GitHub
2. Connect to Hugging Face Spaces
3. Set `HF_TOKEN` secret in Space settings
4. Deploy with auto-sync from GitHub

#### AWS EC2
```bash
# Launch EC2 instance (t3.large or better)
git clone https://github.com/yourusername/finbot.ai.git
cd finbot.ai
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# Set environment variables
export HF_TOKEN=your_token
python main.py
```

#### Google Cloud Run
```bash
gcloud run deploy finbot --source . --allow-unauthenticated
```

### Production Checklist

- [ ] `.env` file is in `.gitignore` ✓
- [ ] `.env.example` has placeholder values ✓
- [ ] No secrets in git history
- [ ] `requirements.txt` is up-to-date
- [ ] Models cached locally (not in git)
- [ ] Logging configured
- [ ] Error handling implemented
- [ ] Rate limiting configured (if applicable)

## Security Notes

⚠️ **IMPORTANT**: Never commit `.env` file with real secrets!

- The `.env` file is excluded via `.gitignore`
- Always use `.env.example` as a template
- Rotate tokens regularly
- Use environment variable secrets in production

## Troubleshooting

### Out of Memory (OOM)
- Reduce `CHUNK_SIZE` in `.env`
- Set `DEVICE=cpu` if using GPU with limited VRAM
- Use a smaller model

### Slow Inference
- Use `DEVICE=cuda` if you have an NVIDIA GPU
- Reduce `TOP_K_DOCUMENTS` for faster retrieval
- Use quantized model versions

### Model Download Issues
- Ensure `HF_TOKEN` is set correctly
- Check internet connection
- Models cached in `~/.cache/huggingface/`

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Support

- 📖 [Documentation](./docs)
- 🐛 [Issue Tracker](https://github.com/yourusername/finbot.ai/issues)
- 💬 [Discussions](https://github.com/yourusername/finbot.ai/discussions)

## Acknowledgments

- [Mistral AI](https://mistral.ai/) for Mistral 7B
- [Hugging Face](https://huggingface.co/) for model hosting
- [OpenAI](https://openai.com/) for optional API integration

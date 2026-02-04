# FinBot Backend Configuration for Streaming

## API Server
```bash
# Development
uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn backend.api:app --host 0.0.0.0 --port 8000 --workers 4 --loop uvloop
```

## Environment Setup

### Using HuggingFace (Default, Recommended)

**Advantages:**
- No API key required
- Free to use
- Models available: Mistral-7B, Llama-2, Phi, etc.

**Setup:**
```bash
cp .env.example .env
# Edit .env:
# USE_OPENAI=false
# DEVICE=cpu (or cuda for GPU)
```

### Using OpenAI

**Advantages:**
- Higher quality responses
- Fastest inference
- GPT-4 available

**Setup:**
```bash
cp .env.example .env
# Edit .env:
# USE_OPENAI=true
# OPENAI_API_KEY=sk_...
```

## Performance Tuning

### CPU Performance
```bash
DEVICE=cpu
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=250
BATCH_SIZE=8
TEMPERATURE=0.7
```

### GPU Performance
```bash
DEVICE=cuda
EMBEDDING_MODEL=all-mpnet-base-v2
CHUNK_SIZE=1000
BATCH_SIZE=32
TOP_K_DOCUMENTS=10
```

## Troubleshooting

### FAISS Index Issues
```bash
# Rebuild index
python -c "from backend.rag import create_rag_pipeline; rag = create_rag_pipeline(); rag.clear_index(); rag.load_documents_from_folder('data/documents')"
```

### Memory Issues
```bash
# Reduce batch size in .env
FINETUNE_BATCH_SIZE=4
CHUNK_SIZE=250
```

### CUDA Issues
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU
DEVICE=cpu
```

## Optional Dependencies

### Fine-Tuning with LoRA

To use the fine-tuning module, install optional dependencies:

```bash
pip install peft==0.7.1 bitsandbytes==0.41.0
```

Or uncomment in `requirements.txt` and reinstall:
```bash
pip install -r requirements.txt
```

**Note:** Fine-tuning is optional. The main API works without these dependencies.

### OpenAI Integration

For using OpenAI models instead of HuggingFace:

```bash
pip install openai==1.3.0
```

Then in `.env`:
```bash
USE_OPENAI=true
OPENAI_API_KEY=sk_...
```

### GPU Support

For faster inference with NVIDIA GPU:

1. Install CUDA from https://developer.nvidia.com/cuda-downloads
2. Install GPU-enabled PyTorch:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
3. Install GPU-optimized FAISS:
   ```bash
   pip install faiss-gpu
   ```

Then set in `.env`:
```bash
DEVICE=cuda
```


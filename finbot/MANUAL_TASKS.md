╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║           WHAT THE USER MUST DO MANUALLY - CHECKLIST              ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝

This document outlines all manual steps required to get FinBot running in your environment.


═══════════════════════════════════════════════════════════════════════════════
1. ENVIRONMENT SETUP (Required)
═══════════════════════════════════════════════════════════════════════════════

□ Install Python 3.10 or higher
  Download from: https://www.python.org/downloads/
  Verify: python --version

□ Install pip (Python package manager)
  Usually included with Python
  Verify: pip --version

□ Create virtual environment
  
  Linux/macOS:
  ```
  python3 -m venv venv
  source venv/bin/activate
  ```
  
  Windows:
  ```
  python -m venv venv
  venv\Scripts\activate
  ```

□ Install project dependencies
  
  ```
  pip install -r requirements.txt
  ```
  
  This installs:
  - fastapi, uvicorn (backend)
  - streamlit (frontend)
  - torch, transformers (LLMs)
  - sentence-transformers, faiss-cpu (embeddings)
  - python-dotenv, pydantic (configuration)
  - PyPDF2 (document processing)
  - Additional supporting libraries

□ Verify installations
  
  ```
  python -c "import fastapi, streamlit, torch, transformers, faiss; print('✓ All imports successful')"
  ```


═══════════════════════════════════════════════════════════════════════════════
2. CONFIGURATION SETUP (Important)
═══════════════════════════════════════════════════════════════════════════════

□ Create .env file from template
  
  ```
  cp .env.example .env
  ```
  
  On Windows:
  ```
  copy .env.example .env
  ```

□ Edit .env file with your settings
  
  Open .env in text editor and configure:
  
  BASIC SETTINGS (can keep defaults):
  ├─ API_HOST=0.0.0.0              (server address)
  ├─ API_PORT=8000                 (backend port)
  ├─ EMBEDDING_MODEL=all-MiniLM-L6-v2 (fast embeddings)
  ├─ CHUNK_SIZE=500                (document chunk size)
  ├─ TOP_K_DOCUMENTS=5             (retrieval count)
  └─ DEVICE=cpu                    (or cuda for GPU)
  
  CHOOSE LLM (one of the following):
  
  Option A - HuggingFace (Default, Recommended)
  ├─ USE_OPENAI=false
  ├─ HF_MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.1
  └─ HF_API_TOKEN=                 (leave empty or get from huggingface.co)
  
  Option B - OpenAI (Requires API key)
  ├─ USE_OPENAI=true
  ├─ OPENAI_API_KEY=sk_XXXXXXXXX   (get from openai.com)
  └─ OPENAI_MODEL=gpt-3.5-turbo

□ If using OpenAI:
  
  1. Visit: https://platform.openai.com/account/api-keys
  2. Create new API key
  3. Copy and paste into OPENAI_API_KEY in .env
  4. Keep this key SECRET and never commit to git

□ If using HuggingFace:
  
  1. (Optional) Visit: https://huggingface.co/settings/tokens
  2. Create access token
  3. Paste into HF_API_TOKEN if using gated models
  4. Large models will auto-download on first run (~7GB for Mistral)


═══════════════════════════════════════════════════════════════════════════════
3. PROJECT STRUCTURE INITIALIZATION (Automatic, Verify)
═══════════════════════════════════════════════════════════════════════════════

The following directories are created automatically, but verify they exist:

□ data/documents/
  Purpose: Place your PDF/TXT files here
  Example files to add:
  ├─ finance_guide.pdf
  ├─ investment_basics.txt
  └─ market_overview.pdf

□ embeddings/
  Purpose: Stores FAISS vector index (auto-created)
  Files generated:
  ├─ faiss_index.index (vector database)
  ├─ faiss_index_metadata.pkl (metadata)
  └─ faiss_index_documents.pkl (document chunks)

□ logs/
  Purpose: Application logs
  Files generated:
  └─ finbot.log (when app runs)

□ config/
  Purpose: Already populated with settings.py

Create if missing:
```
mkdir -p data/documents embeddings logs
```


═══════════════════════════════════════════════════════════════════════════════
4. ADD FINANCIAL DOCUMENTS (Important)
═══════════════════════════════════════════════════════════════════════════════

The chatbot works best with documents. To add them:

□ Prepare documents
  
  Supported formats:
  ├─ PDF files (.pdf)
  └─ Text files (.txt)
  
  Example topics:
  ├─ Financial concepts (compound interest, diversification)
  ├─ Investment guides (stocks, bonds, ETFs)
  ├─ Risk management
  ├─ Retirement planning
  ├─ Market analysis
  └─ Personal finance tips

□ Add documents to data/documents/
  
  Copy your files:
  ```
  cp my_finance_guide.pdf data/documents/
  cp investment_basics.txt data/documents/
  ```

□ (Optional) Use provided sample
  
  The project includes:
  └─ data/financial_qa.json (10 Q&A pairs for fine-tuning)

□ (Optional) Create sample TXT file for testing
  
  Create data/documents/sample.txt with content like:
  ```
  Compound Interest
  ================
  Compound interest is the eighth wonder of the world.
  Those who understand it earn it; those who don't pay it.
  It's calculated as A = P(1 + r/n)^(nt) where...
  ```

Note: Without documents, the chatbot will still work but needs at least 
one document to provide meaningful context. The backend will load 
documents on startup from data/documents/.


═══════════════════════════════════════════════════════════════════════════════
5. RUNNING THE APPLICATION (Three Options)
═══════════════════════════════════════════════════════════════════════════════

IMPORTANT: Ensure virtual environment is ACTIVATED before running!

Activate venv:
  Linux/macOS: source venv/bin/activate
  Windows: venv\Scripts\activate

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OPTION 1: FULL STACK (Backend + Frontend, Recommended for Development)

Step 1 - Start Backend API (Terminal 1)
  ```
  python -m backend.api
  ```
  
  Expected output:
  ├─ Loading embedding model...
  ├─ Initializing RAG pipeline...
  ├─ Loading documents from data/documents
  ├─ INFO: Uvicorn running on http://0.0.0.0:8000
  └─ INFO: Application startup complete
  
  Verify: Open browser to http://localhost:8000/health
  Should show: {"status": "healthy", ...}

Step 2 - Start Frontend (Terminal 2)
  ```
  streamlit run frontend/app.py
  ```
  
  Expected output:
  ├─ Collecting usage statistics
  ├─ You can now view your Streamlit app in your browser
  └─ URL: http://localhost:8501
  
  Browser auto-opens to http://localhost:8501
  If not, visit: http://localhost:8501

Step 3 - Use the application
  ├─ Type question in chat input
  ├─ Click upload to add documents
  ├─ View sources with relevance scores
  └─ Check status in sidebar

Stop: Press Ctrl+C in both terminals

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OPTION 2: API ONLY (Backend Only, for Integration)

Run backend:
  ```
  python -m backend.api
  ```
  
  API endpoints available at:
  ├─ http://localhost:8000/health
  ├─ http://localhost:8000/docs (Swagger UI)
  ├─ http://localhost:8000/chat (POST)
  ├─ http://localhost:8000/status
  └─ http://localhost:8000/upload

Test with curl:
  ```
  curl http://localhost:8000/health
  curl -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"query": "What is diversification?"}'
  ```

Or use Python:
  ```python
  import requests
  response = requests.post(
    "http://localhost:8000/chat",
    json={"query": "What is compound interest?"}
  )
  print(response.json())
  ```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OPTION 3: DOCKER (Containerized, for Production/Deployment)

Requirements:
  ├─ Docker installed (https://www.docker.com/products/docker-desktop)
  └─ Docker daemon running

Run with docker-compose:
  ```
  docker-compose up -d
  ```
  
  This starts:
  ├─ API on http://localhost:8000
  └─ Frontend on http://localhost:8501
  
  View logs:
  ```
  docker-compose logs -f api
  docker-compose logs -f frontend
  ```
  
  Stop:
  ```
  docker-compose down
  ```

Build custom image:
  ```
  docker build -t finbot:latest .
  docker run -p 8000:8000 -p 8501:8501 \
    -e USE_OPENAI=false \
    -v $(pwd)/data:/app/data \
    finbot:latest
  ```


═══════════════════════════════════════════════════════════════════════════════
6. DEPLOYMENT (For Production)
═══════════════════════════════════════════════════════════════════════════════

Choose one platform:

□ Streamlit Cloud (Recommended for Frontend)
  
  1. Push code to GitHub
  2. Go to share.streamlit.io
  3. Create new app from your GitHub repo
  4. Select frontend/app.py as main file
  5. Add secrets (API keys) in app settings
  6. Share your app URL

□ HuggingFace Spaces (Free Backend)
  
  1. Create Space on huggingface.co
  2. Push code to Space
  3. Configure environment variables
  4. App auto-deploys

□ AWS (Scalable)
  
  Lambda + API Gateway: serverless backend
  Elastic Beanstalk: managed backend
  S3 + CloudFront: static frontend
  See deploy.md for details

□ Google Cloud Platform
  
  Cloud Run: serverless backend
  Firebase Hosting: frontend
  Cloud Storage: documents
  See deploy.md for details

□ Docker Hub / Any Container Registry
  
  1. Build image: docker build -t finbot:latest .
  2. Tag: docker tag finbot:latest username/finbot:latest
  3. Push: docker push username/finbot:latest
  4. Deploy: Pull and run on any platform

See deploy.md for detailed instructions for each platform.


═══════════════════════════════════════════════════════════════════════════════
7. OPTIONAL: FINE-TUNING (Advanced)
═══════════════════════════════════════════════════════════════════════════════

To fine-tune the model on financial Q&A:

□ Prepare Q&A dataset
  
  File format: data/financial_qa.json
  
  Structure:
  [
    {
      "question": "What is compound interest?",
      "answer": "Compound interest is...",
      "context": "Financial concepts"
    }
  ]
  
  Note: Sample file is provided in data/financial_qa.json

□ Install fine-tuning dependencies
  
  ```
  pip install peft torch
  ```

□ Run fine-tuning script
  
  ```
  python -c "
  from training.finetune import FineTuner
  fine_tuner = FineTuner(device='cuda')  # or 'cpu'
  fine_tuner.load_model()
  train_loader, val_loader = fine_tuner.prepare_data('data/financial_qa.json')
  history = fine_tuner.fine_tune(train_loader, val_loader, epochs=3)
  fine_tuner.save_model('models/finbot-finetuned')
  "
  ```

□ Use fine-tuned model
  
  Update .env:
  HF_MODEL_NAME=models/finbot-finetuned

Note: Fine-tuning requires GPU and takes 1-2 hours. 
Recommended only for advanced users.


═══════════════════════════════════════════════════════════════════════════════
8. TESTING CHECKLIST (Verification)
═══════════════════════════════════════════════════════════════════════════════

Before considering setup complete, test:

□ Backend Health Check
  
  ```
  curl http://localhost:8000/health
  ```
  Should return: status: "healthy"

□ API Documentation
  
  Visit: http://localhost:8000/docs
  You should see interactive Swagger UI

□ Status Endpoint
  
  ```
  curl http://localhost:8000/status
  ```
  Should show documents_count, device, etc.

□ Frontend Loading
  
  Visit: http://localhost:8501
  Should see chat interface with sidebar

□ Test Chat (With Documents)
  
  1. Upload document (frontend or via API)
  2. Type question: "What is your document about?"
  3. Get response with sources
  
  If no documents:
  Ask: "What is diversification?"
  Should still get generic response

□ Document Upload
  
  Frontend: Use "Upload Documents" in sidebar
  API: curl -X POST http://localhost:8000/upload -F "files=@document.pdf"

□ Source References
  
  After getting response, click "View Sources"
  Should show document names and relevance scores

□ Settings Configuration
  
  Sidebar: Change "Number of Documents" slider
  Submit new question - should retrieve different number of docs

Test Completed ✓


═══════════════════════════════════════════════════════════════════════════════
9. TROUBLESHOOTING GUIDE
═══════════════════════════════════════════════════════════════════════════════

Issue: "ModuleNotFoundError: No module named 'fastapi'"
Solution:
  ├─ Check venv is activated
  ├─ Run: pip install -r requirements.txt
  └─ Verify: pip list | grep fastapi

Issue: "Cannot connect to API at http://localhost:8000"
Solution:
  ├─ Ensure backend is running (python -m backend.api)
  ├─ Check port 8000 is not in use
  ├─ Verify firewall isn't blocking
  └─ Check .env API_PORT setting

Issue: "CUDA out of memory"
Solution:
  ├─ Use CPU instead: DEVICE=cpu in .env
  ├─ Reduce CHUNK_SIZE
  ├─ Reduce FINETUNE_BATCH_SIZE
  └─ Use smaller model: EMBEDDING_MODEL=all-MiniLM-L6-v2

Issue: "Model not found on HuggingFace"
Solution:
  ├─ Check model name: HF_MODEL_NAME
  ├─ Login: huggingface-cli login
  ├─ For gated models, accept terms on HF website

Issue: "No documents in index"
Solution:
  ├─ Add files to data/documents/
  ├─ Check file extensions (.pdf or .txt)
  ├─ Restart backend
  ├─ Check logs: tail logs/finbot.log

Issue: "Slow inference time"
Solution:
  ├─ Use GPU: DEVICE=cuda
  ├─ Reduce TOP_K_DOCUMENTS
  ├─ Use faster model: HF_MODEL_NAME=google/flan-t5-base
  ├─ Reduce response length: MAX_TOKENS=256

Issue: ".env file not found"
Solution:
  ├─ Create from template: cp .env.example .env
  ├─ Check working directory
  ├─ Verify file exists: ls .env or dir .env

For more help, see:
  ├─ README.md - General documentation
  ├─ deploy.md - Deployment issues
  ├─ logs/finbot.log - Application logs
  └─ GitHub issues (if using GitHub)


═══════════════════════════════════════════════════════════════════════════════
10. SECURITY CHECKLIST (Important!)
═══════════════════════════════════════════════════════════════════════════════

□ NEVER commit .env to git
  
  Verify .gitignore contains: .env
  Check: git status (should not show .env)

□ NEVER share API keys
  
  ├─ Keep OPENAI_API_KEY secret
  ├─ Keep HF_API_TOKEN private
  ├─ Use environment variables
  └─ Rotate keys if exposed

□ Use HTTPS in production
  
  ├─ Cloudflare: Free SSL/TLS
  ├─ Let's Encrypt: Free certificates
  └─ AWS ACM: AWS certificate manager

□ Implement authentication
  
  For production, add:
  ├─ API key validation
  ├─ User login (JWT tokens)
  ├─ Rate limiting
  └─ CORS restrictions

□ Keep dependencies updated
  
  ```
  pip install --upgrade -r requirements.txt
  pip audit  # Check for vulnerabilities
  ```

□ Monitor logs regularly
  
  Check logs/finbot.log for:
  ├─ Error patterns
  ├─ Suspicious requests
  ├─ Performance issues
  └─ Failed authentications

□ Backup your data
  
  ├─ embeddings/ (FAISS index)
  ├─ data/documents/ (source files)
  ├─ .env (keep safe)
  └─ Any custom models in models/


═══════════════════════════════════════════════════════════════════════════════
11. PERFORMANCE OPTIMIZATION TIPS
═══════════════════════════════════════════════════════════════════════════════

For Better Quality (Slower):
  ├─ USE_OPENAI=true (or larger HF model)
  ├─ EMBEDDING_MODEL=all-mpnet-base-v2
  ├─ CHUNK_SIZE=1000
  ├─ TOP_K_DOCUMENTS=10
  └─ TEMPERATURE=0.3

For Faster Performance (CPU):
  ├─ USE_OPENAI=false
  ├─ HF_MODEL_NAME=google/flan-t5-base
  ├─ EMBEDDING_MODEL=all-MiniLM-L6-v2
  ├─ CHUNK_SIZE=250
  ├─ TOP_K_DOCUMENTS=3
  └─ TEMPERATURE=0.7

For GPU Acceleration:
  ├─ DEVICE=cuda (requires NVIDIA GPU)
  ├─ Larger models work well
  ├─ Batch processing faster
  └─ Install: pip install torch-cuda (based on CUDA version)

For Low Memory:
  ├─ DEVICE=cpu
  ├─ EMBEDDING_MODEL=all-MiniLM-L6-v2
  ├─ CHUNK_SIZE=250
  ├─ Smaller documents
  └─ BATCH_SIZE=4


═══════════════════════════════════════════════════════════════════════════════
12. MONITORING & LOGGING
═══════════════════════════════════════════════════════════════════════════════

□ Check application logs
  
  ```
  tail -f logs/finbot.log
  ```
  
  Or in Windows:
  ```
  type logs\finbot.log
  ```

□ Monitor API performance
  
  Check /status endpoint:
  ```
  curl http://localhost:8000/status
  ```
  
  Shows:
  ├─ documents_count: Number of indexed documents
  ├─ embedding_dimension: Vector size
  ├─ device: CPU or GPU
  └─ embedding_model: Model being used

□ Check Docker logs
  
  ```
  docker-compose logs -f api
  docker-compose logs -f frontend
  ```

□ Set up monitoring alerts (Production)
  
  ├─ Datadog: https://www.datadoghq.com/
  ├─ New Relic: https://newrelic.com/
  ├─ AWS CloudWatch: https://aws.amazon.com/cloudwatch/
  └─ Google Cloud Monitoring


═══════════════════════════════════════════════════════════════════════════════
SUMMARY - QUICK START COMMANDS
═══════════════════════════════════════════════════════════════════════════════

1. Setup (Run Once):
   python -m venv venv
   source venv/bin/activate              # macOS/Linux
   # or venv\Scripts\activate            # Windows
   pip install -r requirements.txt
   cp .env.example .env
   mkdir -p data/documents

2. Configure:
   Edit .env (set USE_OPENAI, API keys, etc.)

3. Add Documents:
   Copy PDF/TXT files to data/documents/

4. Run (Terminal 1 - Backend):
   source venv/bin/activate
   python -m backend.api

5. Run (Terminal 2 - Frontend):
   source venv/bin/activate
   streamlit run frontend/app.py

6. Access:
   Frontend: http://localhost:8501
   API: http://localhost:8000
   Docs: http://localhost:8000/docs

7. Test:
   curl http://localhost:8000/health


═══════════════════════════════════════════════════════════════════════════════
SUPPORT & RESOURCES
═══════════════════════════════════════════════════════════════════════════════

Documentation:
  ├─ README.md - Comprehensive overview
  ├─ deploy.md - Deployment guides
  ├─ SETUP_GUIDE.md - Configuration details
  └─ PROJECT_SUMMARY.md - Architecture summary

API Documentation (Interactive):
  └─ http://localhost:8000/docs (Swagger UI)

Python Libraries:
  ├─ FastAPI: https://fastapi.tiangolo.com/
  ├─ Streamlit: https://streamlit.io/
  ├─ LangChain: https://langchain.com/
  ├─ HuggingFace: https://huggingface.co/
  └─ FAISS: https://github.com/facebookresearch/faiss

Community:
  ├─ HuggingFace Hub: https://huggingface.co/
  ├─ GitHub: https://github.com/
  └─ Stack Overflow: https://stackoverflow.com/

═══════════════════════════════════════════════════════════════════════════════

FinBot is now ready for use! 🚀

Start with Steps 1-6 above, then refer to this checklist for additional setup.
If you encounter issues, check the Troubleshooting Guide (Section 9).

Good luck! 💰

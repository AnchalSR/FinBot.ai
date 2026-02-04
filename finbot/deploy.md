# FinBot Deployment Guide 🚀

Complete guide for deploying FinBot to production environments.

## Table of Contents

1. [Local Development](#local-development)
2. [Streamlit Cloud](#streamlit-cloud-recommended)
3. [HuggingFace Spaces](#huggingface-spaces)
4. [AWS Deployment](#aws-deployment)
5. [Google Cloud Platform](#google-cloud-platform)
6. [Docker Deployment](#docker-deployment)
7. [Environment Setup](#environment-setup)

---

## Local Development

### Quick Setup

```bash
# 1. Clone and navigate
cd finbot

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your settings

# 5. Add documents (optional)
mkdir -p data/documents
# Add your PDF/TXT files here

# 6. Run backend
python -m backend.api
# API runs on http://localhost:8000

# 7. In another terminal, run frontend
streamlit run frontend/app.py
# UI runs on http://localhost:8501
```

### Verify Setup

```bash
# Check API health
curl http://localhost:8000/health

# Check API docs
# Visit http://localhost:8000/docs

# Check Streamlit UI
# Visit http://localhost:8501
```

---

## Streamlit Cloud (Recommended)

Easiest way to deploy Streamlit frontend.

### Prerequisites

- GitHub account
- Repository with code pushed to GitHub

### Deployment Steps

1. **Prepare Repository**

   Ensure your GitHub repo has this structure:
   ```
   finbot/
   ├── frontend/
   │   └── app.py
   ├── backend/
   ├── config/
   ├── utils/
   ├── requirements.txt
   ├── .env.example
   └── README.md
   ```

2. **Create Streamlit Cloud Account**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"

3. **Deploy**
   - Repository: Select your GitHub repo
   - Branch: `main` (or your branch)
   - File path: `finbot/frontend/app.py`
   - Click "Deploy"

4. **Configure Secrets**

   In Streamlit Cloud dashboard:
   - Go to App menu → Settings → Secrets
   - Add your environment variables:

   ```
   # .streamlit/secrets.toml
   API_URL = "http://your-backend-api.com"
   OPENAI_API_KEY = "sk_..."
   HF_API_TOKEN = "hf_..."
   ```

5. **Update Frontend Code**

   ```python
   # frontend/app.py
   import streamlit as st
   
   API_URL = st.secrets.get("API_URL", "http://localhost:8000")
   OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
   ```

### Hosting Backend

You need a backend API. Options:

1. **Use HuggingFace Spaces Backend** (see below)
2. **Use AWS/GCP Backend** (see below)
3. **Deploy Backend Separately**

```bash
# Deploy FastAPI to Heroku
pip install heroku
heroku create finbot-api
heroku config:set USE_OPENAI=false
git push heroku main
```

---

## HuggingFace Spaces

Deploy both frontend and backend on HuggingFace.

### Create Space

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Create new Space
3. Choose Space type: Docker / Python

### For Python Space (Recommended)

1. **Create app.py**

   ```python
   # app.py
   import subprocess
   import streamlit as st
   from threading import Thread
   
   # Start backend in background
   def start_backend():
       subprocess.Popen(["python", "-m", "uvicorn", "backend.api:app", 
                        "--host", "0.0.0.0", "--port", "7860"])
   
   # Start backend
   start_backend()
   
   # Import frontend
   import sys
   sys.path.insert(0, '.')
   from frontend.app import main
   
   main()
   ```

2. **Create requirements.txt**

   ```
   streamlit==1.29.0
   fastapi==0.104.1
   uvicorn==0.24.0
   # ... other dependencies
   ```

3. **Set Secrets**

   In Space settings → Repository secrets:
   ```
   OPENAI_API_KEY=sk_...
   HF_API_TOKEN=hf_...
   ```

4. **Push Code**

   ```bash
   git clone https://huggingface.co/spaces/username/finbot
   cd finbot
   cp -r ../finbot . 
   git add .
   git commit -m "Add FinBot"
   git push
   ```

### For Docker Space

1. **Create Dockerfile**

   ```dockerfile
   FROM python:3.11-slim
   
   WORKDIR /app
   
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   
   COPY . .
   
   EXPOSE 7860
   
   CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
   ```

2. **Create docker-compose.yml**

   ```yaml
   version: '3.8'
   services:
     api:
       build: .
       ports:
         - "7860:7860"
       environment:
         USE_OPENAI: ${USE_OPENAI}
         OPENAI_API_KEY: ${OPENAI_API_KEY}
       volumes:
         - ./data:/app/data
         - ./embeddings:/app/embeddings
   ```

3. **Push to HuggingFace**

   ```bash
   git clone https://huggingface.co/spaces/username/finbot
   cd finbot
   cp -r ../finbot .
   git add .
   git commit -m "Add Docker FinBot"
   git push
   ```

---

## AWS Deployment

### Deploy Backend to AWS Lambda + API Gateway

#### Using Serverless Framework

1. **Install Serverless**

   ```bash
   npm install -g serverless
   serverless login
   ```

2. **Create serverless.yml**

   ```yaml
   service: finbot-api
   
   provider:
     name: aws
     runtime: python3.11
     region: us-east-1
     environment:
       USE_OPENAI: ${env:USE_OPENAI}
       OPENAI_API_KEY: ${env:OPENAI_API_KEY}
   
   functions:
     api:
       handler: backend.api.app
       events:
         - http:
             path: /{proxy+}
             method: ANY
             cors: true
   
   plugins:
     - serverless-wsgi
     - serverless-python-requirements
   ```

3. **Deploy**

   ```bash
   serverless deploy --param="openaiKey=sk_..."
   ```

#### Using AWS Elastic Beanstalk

1. **Prepare Application**

   ```bash
   # Create .ebextensions/python.config
   option_settings:
     aws:elasticbeanstalk:container:python:
       WSGIPath: backend.api:app
     aws:autoscaling:launchconfiguration:
       InstanceType: t3.medium
   ```

2. **Deploy**

   ```bash
   pip install awsebcli
   eb init -p python-3.11 finbot
   eb create finbot-env
   eb deploy
   ```

3. **Get URL**

   ```bash
   eb open
   ```

### Deploy Frontend to AWS S3 + CloudFront

1. **Build Static Files**

   ```bash
   # Export Streamlit as static
   streamlit run frontend/app.py --logger.level=error
   ```

2. **Upload to S3**

   ```bash
   aws s3 sync build/ s3://my-bucket/finbot/
   ```

3. **Create CloudFront Distribution**

   - Origin: S3 bucket
   - Behavior: Redirect to app.py for routing

### RDS Database (Optional)

For storing chat history:

```bash
# Create RDS instance
aws rds create-db-instance \
  --db-instance-identifier finbot-db \
  --db-instance-class db.t3.micro \
  --engine postgres \
  --master-username admin \
  --master-user-password <password> \
  --allocated-storage 20
```

---

## Google Cloud Platform

### Deploy to Cloud Run

1. **Create app.yaml**

   ```yaml
   runtime: python311
   
   env: standard
   
   entrypoint: uvicorn backend.api:app --host 0.0.0.0 --port $PORT
   
   env_variables:
     USE_OPENAI: "false"
   
   build:
     runtime_version: "3.11"
   ```

2. **Deploy**

   ```bash
   gcloud run deploy finbot-api \
     --source . \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --set-env-vars USE_OPENAI=false,OPENAI_API_KEY=sk_...
   ```

3. **Get URL**

   ```bash
   # Your API will be at:
   # https://finbot-api-{hash}.run.app
   ```

### Deploy Frontend to Firebase Hosting

1. **Install Firebase**

   ```bash
   npm install -g firebase-tools
   firebase login
   ```

2. **Initialize Firebase**

   ```bash
   firebase init hosting
   ```

3. **Update firebase.json**

   ```json
   {
     "hosting": {
       "public": "frontend",
       "rewrites": [
         {
           "source": "**",
           "destination": "/app.py"
         }
       ]
     }
   }
   ```

4. **Deploy**

   ```bash
   firebase deploy
   ```

### Cloud Storage for Documents

```python
from google.cloud import storage

def upload_to_gcs(local_path, bucket_name, blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)

# Usage
upload_to_gcs("data/documents/finance.pdf", "finbot-docs", "finance.pdf")
```

---

## Docker Deployment

### Build Docker Image

1. **Create Dockerfile**

   ```dockerfile
   # Use official Python runtime as base
   FROM python:3.11-slim
   
   WORKDIR /app
   
   # Copy requirements
   COPY requirements.txt .
   
   # Install dependencies
   RUN pip install --no-cache-dir -r requirements.txt
   
   # Copy project
   COPY . .
   
   # Create necessary directories
   RUN mkdir -p logs data/documents embeddings
   
   # Expose ports
   EXPOSE 8000 8501
   
   # Health check
   HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
     CMD curl -f http://localhost:8000/health || exit 1
   
   # Run both services
   CMD ["sh", "-c", "python -m backend.api & streamlit run frontend/app.py"]
   ```

2. **Create .dockerignore**

   ```
   __pycache__
   *.pyc
   *.pyo
   .env
   venv
   .git
   .DS_Store
   ```

3. **Build Image**

   ```bash
   docker build -t finbot:latest .
   ```

4. **Run Container**

   ```bash
   docker run -p 8000:8000 -p 8501:8501 \
     -e USE_OPENAI=false \
     -e OPENAI_API_KEY=sk_... \
     -v $(pwd)/data:/app/data \
     -v $(pwd)/embeddings:/app/embeddings \
     finbot:latest
   ```

### Docker Compose

1. **Create docker-compose.yml**

   ```yaml
   version: '3.8'
   
   services:
     api:
       build: .
       ports:
         - "8000:8000"
       environment:
         - USE_OPENAI=false
         - DEVICE=cpu
       volumes:
         - ./data:/app/data
         - ./embeddings:/app/embeddings
       restart: unless-stopped
     
     frontend:
       build: .
       ports:
         - "8501:8501"
       depends_on:
         - api
       environment:
         - API_URL=http://api:8000
       restart: unless-stopped
   
   volumes:
     data:
     embeddings:
   ```

2. **Run**

   ```bash
   docker-compose up -d
   ```

### Deploy to Docker Hub

```bash
# Tag image
docker tag finbot:latest username/finbot:latest

# Login
docker login

# Push
docker push username/finbot:latest

# Others can pull
docker pull username/finbot:latest
```

---

## Environment Setup

### Production Environment Variables

**.env (Production)**

```bash
# API
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=false           # Set to false for production

# LLM
USE_OPENAI=true
OPENAI_API_KEY=sk_...
OPENAI_MODEL=gpt-4         # Better quality in production

# Embedding
EMBEDDING_MODEL=all-mpnet-base-v2  # Larger, more accurate

# RAG
TOP_K_DOCUMENTS=5
CHUNK_SIZE=1000            # Larger chunks for accuracy

# Performance
DEVICE=cuda                # Use GPU if available
MAX_TOKENS=1024
TEMPERATURE=0.3            # Lower for consistency

# Logging
LOG_LEVEL=INFO
LOG_PATH=/var/log/finbot.log

# Security
ALLOWED_ORIGINS=["https://yourdomain.com"]
API_KEY_SECRET=<secret>    # For API authentication
```

### Secrets Management

#### Option 1: AWS Secrets Manager

```python
import boto3

client = boto3.client('secretsmanager')
secret = client.get_secret_value(SecretId='finbot/prod')
config = json.loads(secret['SecretString'])
```

#### Option 2: Google Secret Manager

```python
from google.cloud import secretmanager

def access_secret_version(secret_id, version_id="latest"):
    client = secretmanager.SecretManagerServiceClient()
    project_id = "your-project"
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")
```

#### Option 3: HashiCorp Vault

```bash
# Start Vault
vault server -dev

# Store secrets
vault kv put secret/finbot openai_key=sk_ hf_token=hf_

# Access from app
import hvac
client = hvac.Client(url='http://localhost:8200', token='...')
secrets = client.secrets.kv.read_secret_version(path='finbot')
```

---

## Monitoring & Observability

### Application Monitoring

#### Using Prometheus + Grafana

```python
# Add to api.py
from prometheus_client import Counter, Histogram, generate_latest

request_count = Counter('chat_requests_total', 'Total chat requests')
request_duration = Histogram('chat_request_duration_seconds', 'Request duration')

@app.get("/metrics")
async def metrics():
    return generate_latest()
```

#### Using DataDog

```python
from datadog import initialize, api
import logging
from datadog_python_logging_handler import DatadogHandler

options = {'api_key': 'YOUR_API_KEY', 'app_key': 'YOUR_APP_KEY'}
initialize(**options)

handler = DatadogHandler()
logging.getLogger().addHandler(handler)
```

### Log Aggregation

#### Using CloudWatch

```python
import boto3
import watchtower
import logging

cloudwatch_handler = watchtower.CloudWatchLogHandler()
logger.addHandler(cloudwatch_handler)
```

#### Using ELK Stack

```yaml
# docker-compose.yml
elasticsearch:
  image: elasticsearch:7.14.0
  
logstash:
  image: logstash:7.14.0
  
kibana:
  image: kibana:7.14.0
```

---

## Database Setup

### PostgreSQL

```bash
# Using managed service (AWS RDS, Google Cloud SQL)
DATABASE_URL=postgresql://user:password@host:5432/finbot

# Or self-hosted
docker run -d \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=finbot \
  -p 5432:5432 \
  postgres:15
```

### Chat History Schema

```sql
CREATE TABLE conversations (
  id SERIAL PRIMARY KEY,
  user_id VARCHAR(255) NOT NULL,
  query TEXT NOT NULL,
  answer TEXT NOT NULL,
  sources JSONB,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_user_id ON conversations(user_id);
```

---

## Performance Optimization

### Caching

```python
# Redis caching for embeddings
import redis

cache = redis.Redis(host='localhost', port=6379, db=0)

def get_embeddings(text):
    # Check cache
    cached = cache.get(f"embedding:{text}")
    if cached:
        return cached
    
    # Generate and cache
    embedding = model.encode(text)
    cache.set(f"embedding:{text}", embedding)
    return embedding
```

### Load Balancing

```yaml
# nginx.conf
upstream api_backend {
  server api1:8000;
  server api2:8000;
  server api3:8000;
}

server {
  listen 80;
  location / {
    proxy_pass http://api_backend;
  }
}
```

---

## Security Checklist

- [ ] Use HTTPS/TLS
- [ ] Implement API authentication
- [ ] Validate all inputs
- [ ] Sanitize responses
- [ ] Use environment variables for secrets
- [ ] Implement rate limiting
- [ ] Add CORS properly
- [ ] Use secure headers
- [ ] Implement logging for audits
- [ ] Regular dependency updates
- [ ] Database encryption
- [ ] Implement backups

---

## Troubleshooting

### API Not Responding

```bash
# Check logs
docker logs finbot-api

# Check port
netstat -tlnp | grep 8000
```

### Out of Memory

```bash
# Increase container memory
docker run -m 4g finbot:latest

# Or in docker-compose.yml
services:
  api:
    deploy:
      resources:
        limits:
          memory: 4G
```

### Database Connection Issues

```bash
# Test connection
psql postgresql://user:pass@host:5432/finbot

# Check credentials in .env
echo $DATABASE_URL
```

---

## Support & Resources

- [Streamlit Docs](https://docs.streamlit.io/)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [AWS Docs](https://docs.aws.amazon.com/)
- [GCP Docs](https://cloud.google.com/docs)
- [Docker Docs](https://docs.docker.com/)

---

**Last Updated:** February 3, 2024

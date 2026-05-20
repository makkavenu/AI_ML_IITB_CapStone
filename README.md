# Multi-Modal AI Agent for Medical, Legal, Vision, and Object-Detection Workflows

This project is a Dockerized research application that combines a **Streamlit chatbot UI**, a **FastAPI backend**, a **LangGraph-based model router**, specialist model tools, AWS S3/DynamoDB request tracking, medical output guardrails, and observability with Prometheus, Grafana, cAdvisor, node-exporter, and CloudWatch Logs.

The current app supports:

- Text-only chat requests.
- Text + 0 to 10 uploaded files per chat request.
- Direct-to-S3 browser uploads using secured presigned PUT URLs.
- Request persistence and chat-session history in DynamoDB.
- Redis-based final-response caching for repeated agent requests.
- Server-Sent Events (SSE) progress streaming from FastAPI to Streamlit.
- Streamlit answer metadata showing the model/tool chain that answered each question.
- Qwen3-32B model or GPT-4o routing.
- Medical specialist model routing for MedGemma, RETFound, Endo-FM, SAM-Med2D, and TotalSegmentator.
- Legal RAG using OpenAI embeddings, Pinecone, and AWS Bedrock Qwen.
- Vision analysis through Qwen-VL.
- Object detection through YOLOv12.
- General input/output guardrails and additional medical-domain output guardrails.
- Local Windows Docker run mode and EC2 Linux run mode.
- Prometheus/Grafana observability, including Redis cache hit/miss metrics.
- CloudWatch Logs forwarding on EC2.

> **Medical safety note:** This is a research system. It is not a clinical device, diagnostic tool, triage system, or treatment-planning system. Do not use its responses as the sole basis for diagnosis, treatment, medication changes, or emergency decisions.

---

## 1. High-Level Architecture

```text
User in Streamlit UI
        │
        │  Text only
        │  └── POST /api/chat/messages
        │
        │  Text + files
        │  ├── POST /api/uploads/presign
        │  ├── PUT files directly to S3 using presigned URLs
        │  └── POST /api/chat/messages with S3 file references
        │
        ▼
FastAPI backend
        │
        ├── validate request
        ├── verify uploaded S3 file references
        ├── write request metadata to DynamoDB
        ├── run input guardrails
        ├── extract safe file context where possible
        ├── scan extracted file text/context
        ├── check Redis response cache
        ├── enqueue async in-process worker task on cache miss
        └── stream progress/final result through SSE
        │
        ▼
LangGraph model router
        │
        ├── Qwen3-32B or GPT-4o decides tool/model routing
        ├── tool_executor injects verified files/images/context
        ├── specialist tool executes
        ├── router may call another tool, for example specialist → MedGemma
        └── final user-facing answer generated
        │
        ▼
Output safety layer
        │
        ├── general output guardrail scan
        ├── medical-domain output guardrails when medical tools are used
        ├── update DynamoDB final status
        ├── store successful final response in Redis
        └── stream final response + answer model name to Streamlit
```

---

## 2. Project Structure

```text
.
├── Dockerfile.api
├── Dockerfile.ui
├── docker-compose.yml
├── docker-compose.ec2-logs.yml
├── requirements.txt
├── requirements-ui.txt
├── pytest.ini
├── .env
├── README.md
├── sample_file.py
│
├── app/
│   ├── main.py
│   ├── guardrails/
│   │   └── guardrail_scanner.py
│   ├── routers/
│   │   ├── chat.py
│   │   ├── uploads.py
│   │   ├── retfound.py
│   │   └── sam_med2d.py
│   └── services/
│       ├── aws_clients.py
│       ├── cache.py
│       ├── dynamodb_store.py
│       ├── event_bus.py
│       ├── file_processing.py
│       ├── medical_output_guardrails.py
│       └── metrics.py
│
├── agent/
│   ├── graph.py
│   └── tools/
│       ├── query_legal_rag.py
│       └── tool_definitions.py
│
├── ui/
│   └── streamlit_app.py
│
├── tests/
│   ├── test_cache_service.py
│   └── test_redis_integration.py
│
└── monitoring/
    ├── README.md
    ├── prometheus/
    │   ├── prometheus.yml
    │   └── prometheus.ec2.yml
    ├── grafana/
    │   ├── provisioning/
    │   │   ├── datasources/
    │   │   │   └── prometheus.yml
    │   │   └── dashboards/
    │   │       └── dashboards.yml
    │   └── dashboards/
    │       └── ai-agent-overview.json
    └── cloudwatch-agent/
        └── amazon-cloudwatch-agent.json
```

---

## 3. Main Components

### 3.1 Streamlit UI

File:

```text
ui/streamlit_app.py
```

Responsibilities:

- Displays the chatbot UI.
- Lets user choose orchestrator model:
  - `qwen3-32b`
  - `gpt-4o`
- Accepts text and up to 10 uploaded files of any format.
- Calls `POST /api/uploads/presign` when files are attached.
- Uploads file bytes directly to S3 through presigned PUT URLs.
- Calls `POST /api/chat/messages` with text, session ID, history, and uploaded S3 references.
- Opens `GET /api/chat/messages/{request_id}/events` for SSE progress/final result.
- Displays tool chain metadata such as `retfound_analyze → medical_qa`.
- Displays the model chain for every assistant answer, for example `Endo-FM → MedGemma 1.5 4B`, `RETFound → MedGemma 1.5 4B`, `Legal RAG (Pinecone + Qwen3-32B)`, or the direct orchestrator model.
- Shows when a response was served from Redis cache.
- Can reload session history from DynamoDB through `GET /api/chat/sessions/{session_id}/messages`.

### 3.2 FastAPI backend

File:

```text
app/main.py
```

Registered routers:

```python
app.include_router(chat_router, prefix="/api")
app.include_router(uploads_router, prefix="/api", tags=["uploads"])
app.include_router(sam_med2d_router, prefix="/api/sam-med2d", tags=["sam-med2d"])
app.include_router(retfound_router, prefix="/api/retfound", tags=["retfound"])
```

Important endpoints:

```text
GET  /health
GET  /metrics
POST /api/uploads/presign
POST /api/chat/messages
GET  /api/chat/messages/{request_id}/events
GET  /api/chat/sessions/{session_id}/messages
POST /api/chat
POST /api/chat/stream
GET  /api/retfound/health
POST /api/retfound/infer
POST /api/sam-med2d/predict
```

### 3.3 LangGraph agent/router

Files:

```text
agent/graph.py
agent/tools/tool_definitions.py
```

The graph uses a ReAct-style loop:

```text
orchestrator → tool_executor → orchestrator → ... → final response
```

The maximum number of tool-executor cycles is controlled in `agent/graph.py`:

```python
MAX_ITERATIONS = 4
```

Supported orchestrator models:

```text
gpt-4o     → OpenAI ChatOpenAI model gpt-4o
qwen3-32b  → AWS Bedrock Converse model qwen.qwen3-32b-v1:0
```

Default orchestrator:

```text
qwen3-32b
```

### 3.4 AWS services

Files:

```text
app/services/aws_clients.py
app/services/dynamodb_store.py
app/services/file_processing.py
```

AWS usage:

- S3 bucket stores uploaded input files.
- DynamoDB table stores request metadata and final results.
- Bedrock is used for Qwen3-32B orchestration and legal RAG answer generation.
- CloudWatch Logs can receive Docker logs on EC2 through `docker-compose.ec2-logs.yml`.

---

### 3.5 Redis response cache

Files:

```text
app/services/cache.py
app/services/metrics.py
docker-compose.yml
```

Caching behavior:

- The API builds a deterministic cache key from the user message, selected orchestrator model, short chat history, and verified uploaded-file metadata.
- Cache lookup happens after input/file guardrails and S3 verification, so unsafe or invalid inputs are not served directly from cache.
- On a Redis hit, the backend updates DynamoDB with `completed_from_cache` and streams the final SSE response with `cache_hit=true`.
- On a miss, the normal LangGraph/tool pipeline runs and successful final responses are stored in Redis with `CACHE_TTL_SECONDS`.
- Prometheus exposes cache hit/miss/set/error/disabled counters and cache get/set latency histograms.

---

## 4. Environment Variables

Create a `.env` file in the project root.

> Do not commit `.env` to GitHub. Rotate any real API keys that were pasted into chat, screenshots, or public documents.

```env
# OpenAI
OPENAI_API_KEY=your_openai_key

# Legal RAG
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX=legal-rag-iitb-project

# External model endpoints
QWEN_VL_ENDPOINT_URL=http://137.74.88.197:8001
YOLOV12_ENDPOINT_URL=http://137.74.88.197:8002
MEDGEMMA_ENDPOINT_URL=http://137.74.88.197:8000
SAM_MED2D_ENDPOINT_URL=http://137.74.88.197:8004
RETFOUND_ENDPOINT_URL=http://137.74.88.197:8005
ENDOFM_ENDPOINT_URL=http://137.74.88.197:8006
TOTALSEG_ENDPOINT_URL=http://137.74.88.197:8003

# Overrides for local FastAPI wrappers used inside the API container
RETFOUND_AGENT_URL=http://127.0.0.1:8000/api/retfound/infer
SAM_MED2D_AGENT_URL=http://127.0.0.1:8000/api/sam-med2d/predict

# HTTP timeout for model/API calls
TOOL_HTTP_TIMEOUT=300

# AWS credentials for local laptop testing only.
# On EC2, prefer an IAM role and omit these two long-term keys.
AWS_ACCESS_KEY_ID=your_access_key_for_local_testing
AWS_SECRET_ACCESS_KEY=your_secret_key_for_local_testing

# AWS regions
# AWS_DEFAULT_REGION is used by Bedrock/Qwen in the router and legal RAG.
AWS_DEFAULT_REGION=us-east-1

# AWS_INDIA_REGION is used by S3 and DynamoDB helper clients.
AWS_INDIA_REGION=ap-south-1

# S3 + DynamoDB
S3_BUCKET_NAME=ai-agent-project-requests-uploads
DDB_TABLE_NAME=ai-agent-requests
DDB_SESSION_INDEX_NAME=session_id-created_at-index

# Redis response cache
REDIS_URL=redis://redis:6379/0
CACHE_ENABLED=true
CACHE_TTL_SECONDS=3600
CACHE_KEY_PREFIX=ai_agent:response

# Observability
PROMETHEUS_RETENTION=7d
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=change_this_password

# EC2 Linux host/container monitoring mode only:
# PROMETHEUS_CONFIG_FILE=./monitoring/prometheus/prometheus.ec2.yml
```

---

## 5. AWS Resources Required

### 5.1 S3 bucket

Expected bucket:

```text
ai-agent-project-requests-uploads
```

Expected prefixes:

```text
inputs/     # user-uploaded request files
outputs/    # future generated output files
```

The application uploads user files to:

```text
inputs/YYYY/MM/DD/<upload_batch_id>/<upload_id>-<filename>
```

### 5.2 S3 CORS for browser/Streamlit uploads

For testing, configure S3 CORS similar to:

```json
[
  {
    "AllowedHeaders": ["*"],
    "AllowedMethods": ["PUT", "GET", "HEAD"],
    "AllowedOrigins": [
      "http://localhost:8501",
      "http://127.0.0.1:8501",
      "http://YOUR_EC2_PUBLIC_IP:8501"
    ],
    "ExposeHeaders": ["ETag"],
    "MaxAgeSeconds": 3000
  }
]
```

Replace `YOUR_EC2_PUBLIC_IP` with your actual EC2 public IP or domain. Do not allow public write permissions on the bucket. File uploads are handled through presigned PUT URLs.

### 5.3 DynamoDB table

Expected table:

```text
ai-agent-requests
```

Primary key:

```text
Partition key: request_id  String
```

Required GSI for session history:

```text
Index name: session_id-created_at-index
Partition key: session_id  String
Sort key: created_at       String
Projection: All
```

The app writes fields such as:

```text
request_id
session_id
message
files
file_count
orchestrator_model
status
stage
created_at
updated_at
history_turn_count
processed_files
response
tool_used
tools_chain
guardrail_flagged
medical_guardrail_applied
medical_guardrail_warnings
medical_guardrail_risk_categories
completed_at
failed_at
error_message
```

---

## 6. IAM Permissions

### 6.1 Local Windows testing

For local Docker on Windows 11, the app reads AWS credentials from `.env`.

Minimum IAM permissions for the IAM user:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "S3AiAgentUploads",
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:HeadObject",
        "s3:AbortMultipartUpload"
      ],
      "Resource": [
        "arn:aws:s3:::ai-agent-project-requests-uploads/inputs/*",
        "arn:aws:s3:::ai-agent-project-requests-uploads/outputs/*",
        "arn:aws:s3:::ai-agent-project-requests-uploads/medical_test/*"
      ]
    },
    {
      "Sid": "DynamoDBAiAgentRequests",
      "Effect": "Allow",
      "Action": [
        "dynamodb:PutItem",
        "dynamodb:UpdateItem",
        "dynamodb:GetItem",
        "dynamodb:Query"
      ],
      "Resource": [
        "arn:aws:dynamodb:ap-south-1:<ACCOUNT_ID>:table/ai-agent-requests",
        "arn:aws:dynamodb:ap-south-1:<ACCOUNT_ID>:table/ai-agent-requests/index/session_id-created_at-index"
      ]
    },
    {
      "Sid": "BedrockQwenAccess",
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": "*"
    }
  ]
}
```

Replace `<ACCOUNT_ID>` with your AWS account ID.

### 6.2 EC2 deployment

On EC2, use an **EC2 IAM role** instead of storing long-term AWS keys in `.env`. Attach permissions for S3, DynamoDB, Bedrock, and CloudWatch Logs.

Additional CloudWatch Logs permissions for EC2 Docker logging:

```json
{
  "Sid": "CloudWatchLogsDocker",
  "Effect": "Allow",
  "Action": [
    "logs:CreateLogGroup",
    "logs:CreateLogStream",
    "logs:PutLogEvents",
    "logs:DescribeLogStreams"
  ],
  "Resource": "*"
}
```

---

## 7. Local Windows 11 Run Steps

### 7.1 Prerequisites

Install:

```text
Docker Desktop
Git or ZIP extraction utility
PowerShell
```

### 7.2 Start the stack

From the project root:

```powershell
docker compose exec redis redis-cli FLUSHDB
docker compose down --remove-orphans
docker compose build --no-cache
docker compose up
```

Default local services:

```text
api        FastAPI backend
ui         Streamlit frontend
redis      Redis response cache
prometheus Prometheus metrics store
grafana    Grafana dashboards
```

The Linux host-monitoring containers are disabled locally by default:

```text
node-exporter
cadvisor
```

They are behind the `linux-host-monitoring` profile because Windows Docker Desktop/WSL2 does not support the same Linux host bind-mount propagation used on EC2.

### 7.3 Local URLs

```text
Streamlit UI:  http://localhost:8501
FastAPI docs:  http://localhost:8000/docs
Health check:  http://localhost:8000/health
Metrics:       http://localhost:8000/metrics
Grafana:       http://localhost:3000
Prometheus:    http://localhost:9090
```

Prometheus is bound to `127.0.0.1:9090` in Docker Compose, so it is local-only. Redis is also bound to `127.0.0.1:6379` for local debugging and is accessed by the API through `redis://redis:6379/0` inside Docker.

### 7.4 Grafana login

```text
Username: value of GRAFANA_ADMIN_USER
Password: value of GRAFANA_ADMIN_PASSWORD
```

Open dashboard:

```text
AI Agent → AI Agent Overview
```

### 7.5 Prometheus targets

Open:

```text
http://localhost:9090/targets
```

Expected local targets:

```text
fastapi-api    UP
prometheus     UP
```

---

## 8. EC2 Deployment Steps

### 8.1 Recommended EC2 instance

For a demo:

```text
AMI: Ubuntu 22.04 or Ubuntu 24.04
Instance type: t3.large or larger
Storage: 30–50 GB gp3
Subnet: public subnet
Public IP: enabled
IAM role: attached
```

### 8.2 Security group inbound rules

Recommended:

| Port | Purpose | Source |
|---:|---|---|
| 22 | SSH | Your IP only |
| 8501 | Streamlit UI | Your IP / demo IP only |
| 3000 | Grafana | Your IP only |
| 8000 | FastAPI direct testing | Your IP only, optional |
| 9090 | Prometheus | Do not expose publicly |
| 9100 | node-exporter | Do not expose publicly |
| 8080 | cAdvisor | Do not expose publicly |

Use SSH tunneling for Prometheus:

```bash
ssh -i your-key.pem -L 9090:localhost:9090 ubuntu@<EC2_PUBLIC_IP>
```

Then open:

```text
http://localhost:9090
```

### 8.3 Install Docker on EC2

```bash
sudo apt update
sudo apt install -y docker.io docker-compose-plugin git unzip
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker ubuntu
```

Log out and log back in after adding `ubuntu` to the Docker group.

### 8.4 Upload or clone project

Example with ZIP:

```bash
unzip AI_ML_IITB_CapStone.zip -d ai-agent
cd ai-agent
```

Example with Git:

```bash
git clone <your-repo-url>
cd <your-project-folder>
```

### 8.5 Create `.env` on EC2

On EC2, omit `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` if you attached an IAM role.

```bash
nano .env
```

Add the environment variables from Section 4.

For full EC2 Linux monitoring, include:

```env
PROMETHEUS_CONFIG_FILE=./monitoring/prometheus/prometheus.ec2.yml
```

### 8.6 Run on EC2 with full Linux monitoring

```bash
docker compose --profile linux-host-monitoring up -d --build
```

### 8.7 Run on EC2 with CloudWatch Logs enabled

```bash
docker compose -f docker-compose.yml -f docker-compose.ec2-logs.yml --profile linux-host-monitoring up -d --build
```

This sends container logs to CloudWatch Logs groups such as:

```text
/ai-agent-demo/api
/ai-agent-demo/ui
/ai-agent-demo/prometheus
/ai-agent-demo/grafana
/ai-agent-demo/node-exporter
/ai-agent-demo/cadvisor
```

### 8.8 Verify EC2 deployment

```bash
docker ps
docker compose logs -f api
curl http://localhost:8000/health
curl http://localhost:8000/metrics
```

Open in browser:

```text
Streamlit: http://<EC2_PUBLIC_IP>:8501
Grafana:   http://<EC2_PUBLIC_IP>:3000
```

Expected EC2 Prometheus targets when using `linux-host-monitoring` profile:

```text
fastapi-api      UP
prometheus       UP
node-exporter    UP
cadvisor         UP
```

---

## 9. API Reference

### 9.1 `GET /health`

Health check.

Response:

```json
{
  "status": "ok"
}
```

### 9.2 `GET /metrics`

Prometheus-compatible metrics endpoint.

Includes generic FastAPI HTTP metrics and app-specific metrics from `app/services/metrics.py`.

### 9.3 `POST /api/uploads/presign`

Used when a chat request has one or more files.

Request:

```json
{
  "session_id": "optional-session-id",
  "files": [
    {
      "filename": "scan.png",
      "content_type": "image/png",
      "size_bytes": 245000
    }
  ]
}
```

Rules:

```text
Minimum files: 1
Maximum files: 10
Maximum size per file metadata validation: 100 MB
Accepted file formats: any format for upload
Actual parsing: image, PDF, text-like files; medical volumes are routed by metadata/extension
Presigned URL expiry: 900 seconds
```

Response:

```json
{
  "upload_batch_id": "uuid",
  "files": [
    {
      "upload_id": "uuid",
      "filename": "scan.png",
      "content_type": "image/png",
      "size_bytes": 245000,
      "s3_bucket": "ai-agent-project-requests-uploads",
      "s3_key": "inputs/YYYY/MM/DD/batch-id/upload-id-scan.png",
      "s3_uri": "s3://ai-agent-project-requests-uploads/inputs/...",
      "presigned_put_url": "https://...",
      "upload_headers": {
        "Content-Type": "image/png",
        "x-amz-meta-original-filename": "scan.png",
        "x-amz-meta-upload-batch-id": "batch-id"
      },
      "public_url": "https://ai-agent-project-requests-uploads.s3.ap-south-1.amazonaws.com/inputs/...",
      "expires_in_seconds": 900
    }
  ]
}
```

The frontend must send the exact returned `upload_headers` during the S3 PUT request.

### 9.4 `POST /api/chat/messages`

Recommended chat-create endpoint.

Request for text only:

```json
{
  "message": "What are common causes of dry cough?",
  "session_id": "optional-session-id",
  "history": [],
  "files": [],
  "orchestrator_model": "qwen3-32b"
}
```

Request for text + files:

```json
{
  "message": "Analyze this medical image for research-demo findings.",
  "session_id": "optional-session-id",
  "history": [
    {"role": "user", "content": "previous question"},
    {"role": "assistant", "content": "previous answer"}
  ],
  "files": [
    {
      "upload_id": "uuid",
      "filename": "scan.png",
      "content_type": "image/png",
      "size_bytes": 245000,
      "s3_bucket": "ai-agent-project-requests-uploads",
      "s3_key": "inputs/YYYY/MM/DD/.../scan.png",
      "s3_uri": "s3://ai-agent-project-requests-uploads/inputs/.../scan.png",
      "public_url": "https://..."
    }
  ],
  "orchestrator_model": "gpt-4o"
}
```

Response:

```json
{
  "request_id": "uuid",
  "session_id": "uuid",
  "status": "ACCEPTED",
  "events_url": "/api/chat/messages/{request_id}/events"
}
```

Backend behavior:

1. Verifies S3 file references using `HeadObject`.
2. Writes initial request metadata to DynamoDB.
3. Runs input text guardrails.
4. Starts async in-process worker using `asyncio.create_task`.
5. Streams progress/final events through SSE.

### 9.5 `GET /api/chat/messages/{request_id}/events`

SSE endpoint for progress/final result.

Example events:

```json
{"type":"accepted","status":"RECEIVED","stage":"received"}
```

```json
{"type":"status","status":"RUNNING","stage":"file_processing"}
```

```json
{"type":"routing","tool":"sam_med2d_segment","message":"I've routed your question to SAM-Med2D segmentation. Processing the request…"}
```

```json
{"type":"tool_done","tool":"sam_med2d_segment"}
```

```json
{
  "type": "final",
  "request_id": "uuid",
  "response": "final answer text",
  "tool_used": "medical_qa",
  "tools_chain": ["sam_med2d_segment", "medical_qa"],
  "guardrail_flagged": false,
  "medical_guardrail_applied": true,
  "medical_guardrail_warnings": [],
  "medical_guardrail_risk_categories": []
}
```

The final SSE event also includes:

```json
{
  "answer_model": "MedGemma 1.5 4B",
  "answer_model_chain": ["Endo-FM", "MedGemma 1.5 4B"],
  "cache_hit": false
}
```

These fields power the Streamlit model-name display and make cached responses auditable.

### 9.6 `GET /api/chat/sessions/{session_id}/messages`

Lists persisted DynamoDB request items for a chat session.

Query parameters:

```text
limit: 1–100, default 50
newest_first: true/false, default true
```

Requires DynamoDB GSI:

```text
session_id-created_at-index
```

Response:

```json
{
  "session_id": "uuid",
  "count": 2,
  "items": [
    {
      "request_id": "uuid",
      "session_id": "uuid",
      "message": "Analyze this image",
      "status": "COMPLETED",
      "response": "..."
    }
  ]
}
```

### 9.7 Legacy endpoints

Still available for backward compatibility:

```text
POST /api/chat
POST /api/chat/stream
```

The current Streamlit UI uses the newer request-ID + SSE flow, not the legacy endpoints.

### 9.8 RETFound wrapper

```text
GET  /api/retfound/health
POST /api/retfound/infer
```

Request:

```json
{
  "request_id": "demo-retfound-001",
  "model": "retfound-cfp",
  "dataset": "HRF",
  "image_url": "https://.../HRF_01_h.jpg",
  "task": "retinal_foundation_embedding",
  "classes": ["healthy", "diabetic_retinopathy", "glaucoma"],
  "return": ["embedding_preview"]
}
```

The wrapper downloads a public image URL, converts it to JPEG, calls the raw RETFound endpoint, and returns a friendly structured response.

### 9.9 SAM-Med2D wrapper

```text
POST /api/sam-med2d/predict
```

Request:

```json
{
  "request_id": "sam-demo-thorax-001",
  "model": "sam-med2d",
  "image_url": "https://.../s0114_111.png",
  "task": "medical_2d_segmentation",
  "target_label": "heart_ventricle_left",
  "prompt_type": "bbox",
  "bbox": [66, 81, 118, 129],
  "reference_mask_url": "https://.../s0114_111_heart_ventricle_left_000.png",
  "return": [
    "predicted_mask",
    "overlay",
    "area_pixels",
    "dice_if_reference_available",
    "iou_if_reference_available"
  ]
}
```

Response includes:

```text
predicted mask base64
overlay base64
area_pixels
Dice score if reference mask is available
IoU score if reference mask is available
explanation_for_professor
```

---

## 10. Tool and Model Routing

Available tools:

| Tool | Backing model/service | When used |
|---|---|---|
| `medical_qa` | MedGemma 1.5 4B | Medical text and general medical image explanation |
| `retfound_analyze` | RETFound | Retina/fundus/OCT images |
| `endofm_analyze` | Endo-FM | Endoscopy/colonoscopy/capsule-endoscopy/polyp frames or videos |
| `sam_med2d_segment` | SAM-Med2D | 2D medical segmentation with bbox prompt |
| `totalsegmentator_segment` | TotalSegmentator | 3D CT/MR NIfTI/DICOM organ/vessel/bone segmentation |
| `legal_qa` | Pinecone + OpenAI embeddings + Bedrock Qwen | Legal questions |
| `vision_llm` | Qwen-VL endpoint | General image/file visual analysis |
| `object_detection` | YOLOv12 endpoint | Detection/counting/localization in first image |

Medical routing policy:

```text
Only medical text
→ medical_qa / MedGemma

Retina / fundus / OCT image
→ retfound_analyze
→ medical_qa final explanation

Endoscopy / colonoscopy / capsule endoscopy / polyp image/video
→ endofm_analyze
→ medical_qa final explanation

2D medical segmentation needed
→ sam_med2d_segment
→ medical_qa final explanation

3D CT/MR volume such as NIfTI or DICOM
→ totalsegmentator_segment
→ medical_qa final explanation

General medical image + question
→ medical_qa
---

## 11. Guardrails and Safety

### 11.1 General guardrails

File:

```text
app/guardrails/guardrail_scanner.py
```

The general scanner detects:

```text
prompt injection
harmful content
```

Guardrail phases:

```text
input text guardrail
uploaded file context guardrail
output text guardrail
```

### 11.2 S3 file verification

Before accepting a file-backed chat request, FastAPI verifies:

```text
S3 bucket matches configured bucket
S3 key starts with inputs/
S3 object exists
object is not empty
size matches frontend-provided size when supplied
content type is consistent when available
```

### 11.3 Uploaded file context guardrail

File:

```text
app/services/file_processing.py
```

For supported files:

```text
Images → FastAPI loads image base64 in memory for model call; DynamoDB stores only metadata, not base64
PDFs → first pages text extracted with pypdf
Text-like files → UTF-8 text preview extracted
Medical volumes / videos / unknown files → metadata only
```

Extracted text/context is treated as untrusted evidence and scanned before routing.

### 11.4 Medical output guardrails

File:

```text
app/services/medical_output_guardrails.py
```

Applied when any medical tool appears in the tool chain:

```text
medical_qa
retfound_analyze
endofm_analyze
sam_med2d_segment
totalsegmentator_segment
```

Checks include:

```text
missing research/non-diagnostic disclaimer
over-confident diagnosis language
unsafe treatment/medication directives
emergency red-flag terms
specialist model used without MedGemma final explanation
empty medical response
```

The guardrail may add:

```text
Medical safety note
Emergency warning
Safety limitations paragraph
```

---

## 12. Observability

### 12.1 FastAPI metrics

FastAPI exposes:

```text
GET /metrics
```

Generic HTTP metrics are added by `prometheus-fastapi-instrumentator`.

App-specific metrics are in:

```text
app/services/metrics.py
```

Important custom metrics:

```text
ai_agent_chat_requests_total
ai_agent_chat_request_status_total
ai_agent_chat_request_duration_seconds
ai_agent_file_presign_requests_total
ai_agent_file_presign_files_total
ai_agent_guardrail_blocked_total
ai_agent_tool_selection_total
ai_agent_medical_output_guardrail_total
ai_agent_cache_events_total
ai_agent_cache_operation_duration_seconds
```

### 12.2 Prometheus

Local config:

```text
monitoring/prometheus/prometheus.yml
```

Scrapes:

```text
api:8000/metrics
prometheus:9090
```

EC2 config:

```text
monitoring/prometheus/prometheus.ec2.yml
```

Scrapes:

```text
api:8000/metrics
prometheus:9090
node-exporter:9100
cadvisor:8080
```

### 12.3 Grafana

Provisioning files:

```text
monitoring/grafana/provisioning/datasources/prometheus.yml
monitoring/grafana/provisioning/dashboards/dashboards.yml
```

Dashboard:

```text
monitoring/grafana/dashboards/ai-agent-overview.json
```

The dashboard includes panels for chat volume, guardrails, tool routing, worker latency, Redis cache hit/miss/set/error events, and Redis cache get/set P95 latency.

### 12.4 CloudWatch Logs

EC2-only Compose override:

```text
docker-compose.ec2-logs.yml
```

Run with:

```bash
docker compose -f docker-compose.yml -f docker-compose.ec2-logs.yml --profile linux-host-monitoring up -d --build
```

---

## 13. Testing and AI Agent Quality Pipeline

Testing is part of the agent-building pipeline because LLM and agentic systems can fail at multiple layers: request validation, routing, tool invocation, guardrails, caching, persistence, streaming, and UI rendering. This project now includes lightweight automated tests plus a real-Redis integration test.

Test files:

```text
tests/test_cache_service.py
tests/test_redis_integration.py
pytest.ini
```

Run unit tests without Redis:

```bash
pytest -q tests/test_cache_service.py
```

Run all local tests; the real Redis integration test is skipped unless explicitly enabled:

```bash
pytest -q tests
```

Run the Redis integration test after Redis is running locally or through Docker Compose:

```bash
RUN_REDIS_INTEGRATION_TESTS=1 REDIS_URL=redis://localhost:6379/0 pytest -q tests/test_redis_integration.py
```

Testing coverage currently includes:

- Unit tests for deterministic Redis cache-key generation.
- Unit tests for Redis get/set behavior using a fake Redis client.
- Optional integration test against a real Redis service.
- Syntax checks with `python -m py_compile $(find app agent ui -type f -name '*.py') sample_file.py`.
---

## 14. Development Without Docker

API:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows PowerShell
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

UI:

```bash
pip install -r requirements-ui.txt
API_URL=http://localhost:8000 streamlit run ui/streamlit_app.py
```

---

## 15. Quick Command Summary

### Tests

```bash
pytest -q tests
```

### Local Windows

```powershell
docker compose down --remove-orphans
docker compose build --no-cache
docker compose up
```

### EC2 Linux full monitoring

```bash
docker compose --profile linux-host-monitoring up -d --build
```

### EC2 Linux with CloudWatch logs

```bash
docker compose -f docker-compose.yml -f docker-compose.ec2-logs.yml --profile linux-host-monitoring up -d --build
```

### Check logs

```bash
docker compose logs -f api
docker compose logs -f ui
```

### Stop

```bash
docker compose down
```

### Stop and remove volumes

```bash
docker compose down -v
```

---

# AI Server — Qwen 2.5 3B MLX Inference Server

A self-hosted AI inference server running **Qwen 2.5 3B (4-bit quantized)** via Apple's MLX framework on Apple Silicon, exposed publicly through a Cloudflare Tunnel.

---

## Table of Contents

1. [Requirements](#1-requirements)
2. [Repository Structure](#2-repository-structure)
3. [Setup on a New Machine](#3-setup-on-a-new-machine)
4. [Environment Variables & Configuration](#4-environment-variables--configuration)
5. [Running the Server](#5-running-the-server)
6. [Auto-Start on Boot (launchctl)](#6-auto-start-on-boot-launchctl)
7. [Cloudflare Tunnel Setup](#7-cloudflare-tunnel-setup)
8. [API Reference](#8-api-reference)
9. [Training Data & Fine-Tuning](#9-training-data--fine-tuning)
10. [Commands Cheat Sheet](#10-commands-cheat-sheet)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Requirements

| Requirement | Version |
|---|---|
| macOS | 14 Sonoma or later (Apple Silicon) |
| Python | 3.11+ (3.14 recommended) |
| Homebrew | latest |
| cloudflared | latest (via Homebrew) |
| Disk space | ~3 GB for model weights |
| RAM | 8 GB minimum, 16 GB recommended |

> **Apple Silicon only.** MLX is an Apple-specific ML framework (M1/M2/M3/M4). It will not run on Intel Macs or Linux without modifications.

---

## 2. Repository Structure

```
closot-ai/
├── app.py                          # FastAPI inference server (main entry point)
├── .env                 # Environment variable template
├── launchagents/
│   ├── com.mac.mlx-fastapi.plist   # launchctl service for the MLX server
│   └── com.cloudflare.cloudflared.plist  # launchctl service for Cloudflare tunnel
└── cloudflared/
    └── config.yml.example          # Cloudflare tunnel config template
├── requirements.txt
```


---

## 3. Setup on a New Machine

### Step 1 — Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/ai-server.git
cd ai-server
```

### Step 2 — Install Python

```bash
# Install Python 3.11+ via Homebrew if not already installed
brew install python@3.11

# Verify
python3.11 --version
```

### Step 3 — Create and activate a virtual environment

```bash
# From inside the repo directory
python3.11 -m venv venv

# Activate (you must do this every time you open a new terminal)
source venv/bin/activate

# Your prompt should now show (venv) at the start
# To deactivate when done: deactivate
```

### Step 4 — Install dependencies

```bash
# Make sure venv is active first (see step above)
pip install --upgrade pip
pip install -r requirements.txt

# Verify mlx installed correctly
python -c "import mlx.core as mx; print('MLX device:', mx.default_device())"
# Should print: MLX device: Device(gpu, 0)
```

### Step 5 — Download the model

The model weights are not in git. Download using one of these methods:

**Option A — Using mlx_lm (recommended, converts automatically):**
```bash
# Download and convert Qwen2.5-3B to 4-bit quantized MLX format
python -m mlx_lm.convert \
  --hf-path Qwen/Qwen2.5-3B-Instruct \
  --mlx-path qwen25-3b-4bit \
  --quantize \
  --q-bits 4
```

**Option B — Download pre-converted model from HuggingFace:**
```bash
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='mlx-community/Qwen2.5-3B-Instruct-4bit',
    local_dir='qwen25-3b-4bit'
)
"
```

**Option C — Copy from another machine:**
```bash
# From another machine that already has the model:
scp -r /path/to/ai-server/qwen25-3b-4bit user@newmachine:~/ai-server/
```

### Step 6 — Configure environment

```bash
cp .env.example .env
# Open and set your API key
nano .env
# or: echo "LLM_API_KEY=$(openssl rand -hex 32)" > .env
```

### Step 7 — Start the server with uvicorn

```bash
# Make sure venv is active
source venv/bin/activate

# Load your API key and start
export $(cat .env | xargs)
uvicorn app:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 300

# Server is now running at http://0.0.0.0:8000
# Test it:
curl -X POST http://localhost:8000/generate-stream \
  -H "Authorization: $LLM_API_KEY" \
  -d "Hello, who are you?"
```

---

## 4. Environment Variables & Configuration

### `.env` file (gitignored)

Create `.env` in the root of the repo:

```env
LLM_API_KEY=your-secret-api-key-here
```

Generate a strong key:
```bash
openssl rand -hex 32
```

### Set API key for `app.py` and launchctl services

The API key must be set in three places so it's available to `app.py` directly, and to both launchctl services:

**1. In your `.env` file** (used when running manually):
```bash
echo 'LLM_API_KEY=your-api-key-here' > .env
```

**2. In the launchctl environment** (required for launchctl-managed services to pick it up):
```bash
launchctl setenv LLM_API_KEY "your-api-key-here"
```
> Run this once after each reboot, or add it to your shell profile (`~/.zshrc`) so it persists. Without this, the launchctl services won't have access to the key even if it's in `.env`.

**3. In the plist `EnvironmentVariables` block** (persistent across reboots without needing the above):
Edit `~/Library/LaunchAgents/com.mac.mlx-fastapi.plist` and set:
```xml
<key>LLM_API_KEY</key>
<string>your-api-key-here</string>
```
Then reload: `launchctl unload ~/Library/LaunchAgents/com.mac.mlx-fastapi.plist && launchctl load ~/Library/LaunchAgents/com.mac.mlx-fastapi.plist`

### Where the config lives

| Item | Location |
|---|---|
| API key | `ai-server/.env` → `LLM_API_KEY` env var |
| Model path | hardcoded as `qwen25-3b-4bit` in `app.py:25` |
| Server port | `8000` (set in `app.py:52` and launchctl plist) |
| Cloudflare config | `~/.cloudflared/config.yml` |
| Cloudflare credentials | `~/.cloudflared/<tunnel-uuid>.json` (gitignored) |
| MLX server logs | `ai-server/mlx-server.log` / `.err` |
| Cloudflare logs | `~/.cloudflared/tunnel.log` / `tunnel-error.log` |

### Model configuration (`app.py`)

| Parameter | Value | Description |
|---|---|---|
| `temp` | `0.3` | Sampling temperature (lower = more deterministic) |
| `top_p` | `0.95` | Nucleus sampling threshold |
| `min_p` | `0.06` | Minimum probability filter |
| `max_tokens` | `32768` | Maximum output tokens |

---

## 5. Running the Server

### Manual start (development)

```bash
cd closot-ai
source venv/bin/activate
export $(cat .env | xargs)   # loads LLM_API_KEY from .env

# Run via uvicorn (recommended — same as launchctl service)
uvicorn app:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 300

# With --reload for live code changes during development
uvicorn app:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 300 --reload
```

### Test the server is running

```bash
# Open Swagger UI in browser
open http://localhost:8000/docs

# Test inference via curl
curl -X POST http://localhost:8000/generate-stream \
  -H "Authorization: $LLM_API_KEY" \
  -d "What is the capital of France?"
```

---

## 6. Auto-Start on Boot (launchctl)

This sets up the MLX server and Cloudflare tunnel to start automatically on login.

### Step 0 — Set the API key in the launchctl environment

This must be done before loading any services, and after every reboot:

```bash
launchctl setenv LLM_API_KEY "your-api-key-here"
```

To avoid running this after every reboot, add it to `~/.zshrc`:

```bash
echo 'launchctl setenv LLM_API_KEY "your-api-key-here"' >> ~/.zshrc
```

### Install MLX FastAPI service

```bash
# 1. Copy the plist template
cp launchagents/com.mac.mlx-fastapi.plist ~/Library/LaunchAgents/

# 2. Edit it — replace /YOUR_HOME with your actual home directory (e.g. /Users/yourname)
#    and set your actual API key in the EnvironmentVariables block
nano ~/Library/LaunchAgents/com.mac.mlx-fastapi.plist

# 3. Set the API key in the launchctl environment
launchctl setenv LLM_API_KEY "your-api-key-here"

# 4. Bootstrap (load) the service
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.mac.mlx-fastapi.plist

# 5. Start it immediately
launchctl kickstart gui/$(id -u)/com.mac.mlx-fastapi
```

### Install Cloudflare tunnel service

```bash
# 1. Copy the plist template
cp launchagents/com.cloudflare.cloudflared.plist ~/Library/LaunchAgents/

# 2. Edit it — replace /YOUR_HOME with your actual home directory
nano ~/Library/LaunchAgents/com.cloudflare.cloudflared.plist

# 3. Bootstrap (load) the service
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.cloudflare.cloudflared.plist

# 4. Start it immediately
launchctl kickstart gui/$(id -u)/com.cloudflare.cloudflared
```

### launchctl management commands

```bash
# Check service status
launchctl list | grep mlx
launchctl list | grep cloudflare

# Start a service
launchctl kickstart gui/$(id -u)/com.mac.mlx-fastapi
launchctl kickstart gui/$(id -u)/com.cloudflare.cloudflared

# Stop a service
sudo launchctl bootout gui/$(id -u)/com.mac.mlx-fastapi
sudo launchctl bootout gui/$(id -u)/com.cloudflare.cloudflared

# Reload after editing the plist (bootout then bootstrap)
sudo launchctl bootout gui/$(id -u)/com.mac.mlx-fastapi
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.mac.mlx-fastapi.plist
launchctl kickstart gui/$(id -u)/com.mac.mlx-fastapi

# Set env var so running services can read it (do this after every reboot)
launchctl setenv LLM_API_KEY "your-api-key-here"

# View logs
tail -f ~/closot-ai/mlx-server.log
tail -f ~/closot-ai/mlx-server.err
tail -f ~/.cloudflared/tunnel.log
```

---

## 7. Cloudflare Tunnel Setup

This exposes `localhost:8000` publicly at your domain (e.g. `ai.yourdomain.com`) without opening firewall ports.

### Step 1 — Install cloudflared

```bash
brew install cloudflared
```

### Step 2 — Login to Cloudflare

```bash
cloudflared tunnel login
# This opens a browser — authorize your domain
# Creates: ~/.cloudflared/cert.pem
```

### Step 3 — Create a tunnel

```bash
cloudflared tunnel create ai-server
# Output: Created tunnel ai-server with id <YOUR_TUNNEL_UUID>
# Creates: ~/.cloudflared/<YOUR_TUNNEL_UUID>.json
```

### Step 4 — Create the config file

```bash
mkdir -p ~/.cloudflared
cp cloudflared/config.yml.example ~/.cloudflared/config.yml
nano ~/.cloudflared/config.yml
```

Fill in your tunnel UUID and domain:

```yaml
tunnel: YOUR_TUNNEL_UUID_HERE
credentials-file: ~/.cloudflared/YOUR_TUNNEL_UUID_HERE.json

ingress:
  - hostname: ai.yourdomain.com
    service: http://localhost:8000
  - service: http_status:404
```

### Step 5 — Add DNS record

```bash
cloudflared tunnel route dns ai-server ai.yourdomain.com
# This creates a CNAME record in your Cloudflare DNS automatically
```

### Step 6 — Test the tunnel

```bash
# Make sure ai-server is running first (port 8000)
cloudflared tunnel run ai-server

# In another terminal, test:
curl -X POST https://ai.yourdomain.com/generate-stream \
  -H "Authorization: your-api-key" \
  -d "Hello, how are you?"
```

### Step 7 — Set up auto-start (see Section 6)

Once confirmed working, install the launchctl service so the tunnel starts on boot.

---

## 8. API Reference

### Base URL

- Local: `http://localhost:8000`
- Public: `https://ai.yourdomain.com` (via Cloudflare tunnel)

### Authentication

All endpoints require an `Authorization` header with your API key:

```
Authorization: your-api-key-here
```

### Endpoints

#### `POST /generate-stream`

Generate a response from the model.

**Request:**
- Body: raw plain text (the user's prompt)
- Content-Type: `text/plain` (or any — body is read raw)
- Header: `Authorization: <api-key>`

**Response:**
- Content-Type: `text/plain`
- Body: generated text response

**Example:**
```bash
curl -X POST http://localhost:8000/generate-stream \
  -H "Authorization: your-api-key" \
  -d "Explain quantum computing in simple terms"
```

**Python example:**
```python
import requests

response = requests.post(
    "http://localhost:8000/generate-stream",
    headers={"Authorization": "your-api-key"},
    data="What is the meaning of life?"
)
print(response.text)
```

#### `GET /docs`

FastAPI auto-generated Swagger UI (no auth required).

#### `GET /openapi.json`

OpenAPI schema.

---


---

## 9. Commands Cheat Sheet

### Server management

```bash
# Start server (manual, dev)
cd ~/closot-ai && source venv/bin/activate && export $(cat .env | xargs)
uvicorn app:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 300

# Start via launchctl (production, auto-restart)
launchctl start com.mac.mlx-fastapi

# Stop via launchctl
launchctl stop com.mac.mlx-fastapi

# View live logs
tail -f ~/closot-ai/mlx-server.log
tail -f ~/closot-ai/mlx-server.err

# Check if port 8000 is in use
lsof -i :8000
```

### Cloudflare tunnel

```bash
# Start tunnel (manual)
cloudflared tunnel run ai-server

# Start via launchctl
launchctl start com.cloudflare.cloudflared

# Stop via launchctl
launchctl stop com.cloudflare.cloudflared

# View tunnel logs
tail -f ~/.cloudflared/tunnel.log

# List all tunnels
cloudflared tunnel list

# Check tunnel status
cloudflared tunnel info ai-server
```

### Model management

```bash
# Download/convert model
python -m mlx_lm.convert \
  --hf-path Qwen/Qwen2.5-3B-Instruct \
  --mlx-path qwen25-3b-4bit \
  --quantize --q-bits 4

# Run model interactively (no server)
python -m mlx_lm.generate \
  --model qwen25-3b-4bit \
  --prompt "Hello, who are you?"

# Fine-tune
python -m mlx_lm.lora --model qwen25-3b-4bit --train --data . --iters 1000
```

### Python environment

```bash
# Activate virtualenv
source ~/ai-server/venv/bin/activate

# Deactivate
deactivate

# Reinstall all deps
pip install -r requirements.txt

# Check installed versions
pip list | grep mlx
```

### API test calls

```bash
# Basic inference
curl -X POST http://localhost:8000/generate-stream \
  -H "Authorization: $LLM_API_KEY" \
  -d "Your prompt here"

# Via public Cloudflare URL
curl -X POST https://ai.yourdomain.com/generate-stream \
  -H "Authorization: $LLM_API_KEY" \
  -d "Your prompt here"
```

---

## 11. Troubleshooting

### Server won't start

```bash
# Check for errors
cat ~/ai-server/mlx-server.err | tail -50

# Check port conflict
lsof -i :8000

# Run manually to see errors in terminal
cd ~/ai-server && source venv/bin/activate && python app.py
```

### Model loading fails

```bash
# Verify model files exist
ls -lh ~/ai-server/qwen25-3b-4bit/

# Check disk space
df -h ~

# Re-download model
python -m mlx_lm.convert --hf-path Qwen/Qwen2.5-3B-Instruct --mlx-path qwen25-3b-4bit --quantize --q-bits 4
```

### Cloudflare tunnel not connecting

```bash
# View error logs
cat ~/.cloudflared/tunnel-error.log | tail -50

# Check tunnel is authenticated
cloudflared tunnel list

# Re-login if needed
cloudflared tunnel login

# Test tunnel directly
cloudflared tunnel run --config ~/.cloudflared/config.yml ai-server
```

### 401 Unauthorized errors

- Verify the `Authorization` header matches `LLM_API_KEY` exactly
- Check the plist file has the correct API key in `EnvironmentVariables`
- Reload the launchctl service after changing the key:
  ```bash
  launchctl unload ~/Library/LaunchAgents/com.mac.mlx-fastapi.plist
  launchctl load   ~/Library/LaunchAgents/com.mac.mlx-fastapi.plist
  ```

### MLX / Metal GPU issues

```bash
# Check MLX can see GPU
python -c "import mlx.core as mx; print(mx.default_device())"
# Should output: Device(gpu, 0)

# Update MLX
pip install --upgrade mlx mlx-lm mlx-metal
```

---

## Model Details

| Property | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-3B-Instruct` |
| Quantization | 4-bit affine, group_size=64 |
| Parameters | 3 billion |
| Context window | 32,768 tokens |
| Framework | Apple MLX |
| Model size on disk | ~1.6 GB |
| License | Qwen Research License |

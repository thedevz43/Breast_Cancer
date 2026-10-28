# 🚀 Deployment Guide: Deploying BreastVision & M6 on runs-on.dev

This guide provides step-by-step instructions for deploying both the **BreastVision Web / Mobile App** and the **M6 PyTorch Clinical Decision-Support Backend** to **[runs-on.dev](https://runs-on.dev/)**.

---

## 📌 1. How runs-on.dev Works

[runs-on.dev](https://runs-on.dev/) is an open-source, free developer subdomain registry operated by Advance Labs:
* Each subdomain is defined by a JSON file (`domains/<name>.json`) in the public repository [zordhalo/runs-on.dev](https://github.com/zordhalo/runs-on.dev).
* **runs-on.dev handles DNS and HTTPS wildcard routing**, while your code runs on your chosen hosting provider (such as [Render](https://render.com), [Vercel](https://vercel.com), [Railway](https://railway.app), or your own VPS).
* You can configure both your root domain (e.g. `breastvision.runs-on.dev`) and subdomains (e.g. `api.breastvision.runs-on.dev`).

---

## 🏗️ 2. Platform Architecture

We have prepared two deployment configurations in the [`deploy/`](./deploy/) directory:

### Architecture: All-in-One Deployment (Recommended)
Both the web interfaces and the PyTorch FastAPI inference engine run inside the unified Docker container:
* **`https://breastvision.runs-on.dev/`** — M6 Clinical Decision Support Web Prototype
* **`https://breastvision.runs-on.dev/mobile`** — BreastVision React Native / Expo Web Client
* **`https://breastvision.runs-on.dev/api/predict`** — PyTorch Mobile U-Net v2 Inference & Platt Calibration API
* **`https://breastvision.runs-on.dev/api/health`** — System Health & Checkpoint Integrity Manifest

```
                         [ Patient / Clinician Browser ]
                                       │
                                       ▼ HTTPS
                           breastvision.runs-on.dev
                             (Advance Labs DNS zone)
                                       │
                                       ▼ CNAME
                          [ Hosting Provider: Render ]
                         ┌─────────────────────────────┐
                         │   Docker Container          │
                         │ ├─ / (M6 Prototype Web UI)  │
                         │ ├─ /mobile (Expo Web Build) │
                         │ └─ /api/predict (FastAPI)   │
                         └─────────────────────────────┘
```

---

## 📋 3. Step-by-Step Deployment Instructions

### Step 1: Deploy the Container to a Free Cloud Host (e.g., Render)

1. Push your repository to GitHub:
   ```bash
   git add .
   git commit -m "feat: add production deployment and runs-on.dev configs"
   git push origin main
   ```

2. Open **[dashboard.render.com](https://dashboard.render.com/)** (or Railway / Fly.io):
   * Click **New +** → **Web Service**.
   * Connect your GitHub repository: `https://github.com/thedevz43/Breast_Cancer`.
   * Configure the service:
     * **Name**: `breastvision` (or `breastvision-m6`)
     * **Environment**: `Docker`
     * **Dockerfile Path**: `./deploy/Dockerfile`
     * **Docker Context**: `.`
     * **Instance Type**: `Free`
   * Click **Create Web Service**.

3. Once built and healthy, Render will assign you a live URL, for example:
   ```
   breastvision.onrender.com
   ```

---

### Step 2: Claim Your Subdomain on runs-on.dev

You can claim your name either via the runs-on.dev web interface or via a GitHub Pull Request:

#### Option A: Via the Web Interface (Instant)
1. Go to **[https://runs-on.dev](https://runs-on.dev/)** and click **Sign in with GitHub**.
2. Go to **[https://runs-on.dev/manage](https://runs-on.dev/manage)**.
3. Enter your desired subdomain (e.g., `breastvision`).
4. Select record type **CNAME** and paste your Render URL (without `https://`):
   ```
   breastvision.onrender.com
   ```
5. Click **Save Record**. DNS will propagate in seconds.

#### Option B: Via GitHub Pull Request (Open Source)
1. Fork the registry repository: [github.com/zordhalo/runs-on.dev](https://github.com/zordhalo/runs-on.dev).
2. Use the pre-generated record file located at [`deploy/runs-on.dev/domains/breastvision.json`](./deploy/runs-on.dev/domains/breastvision.json):
   ```json
   {
     "name": "breastvision",
     "owner": {
       "github": "thedevz43"
     },
     "claimedAt": "2026-10-28T15:45:00.000Z",
     "records": {
       "CNAME": "breastvision.onrender.com"
     }
   }
   ```
3. Commit this file into the `domains/` folder: `domains/breastvision.json`.
4. Submit a Pull Request. Once CI validates your record, it merges and syncs to DNS automatically.

---

### Step 3: Add Custom Domain to Render

1. Return to your Render Dashboard → Service **breastvision** → **Settings** → **Custom Domains**.
2. Click **Add Custom Domain** and enter:
   ```
   breastvision.runs-on.dev
   ```
3. Render will verify the CNAME record from runs-on.dev and automatically provision an SSL/TLS certificate via Let's Encrypt.
4. Once verified, Render displays a green badge: **"Verified"**.

---

### Step 4: Verify Your Deployment

Open your browser and navigate to:
* **Main Prototype UI**: `https://breastvision.runs-on.dev/`
* **Mobile Web Client**: `https://breastvision.runs-on.dev/mobile`
* **Backend Health**: `https://breastvision.runs-on.dev/health`
* **API Predict**: `https://breastvision.runs-on.dev/api/predict`

---

## 🛠️ 4. Dual-Service Setup (Frontend on Vercel + Backend on Render)

If you prefer to host the Frontend on Vercel and the AI Inference Engine on Render:

1. Deploy the backend to Render (`breastvision-api.onrender.com`).
2. Deploy the frontend to Vercel (`breastvision-web.vercel.app`).
3. In `domains/breastvision.json`, configure both:
   ```json
   {
     "name": "breastvision",
     "owner": {
       "github": "thedevz43"
     },
     "claimedAt": "2026-10-28T15:45:00.000Z",
     "records": {
       "CNAME": "cname.vercel-dns.com"
     },
     "subdomains": {
       "api": {
         "CNAME": "breastvision-api.onrender.com"
       }
     }
   }
   ```
4. Now:
   * `https://breastvision.runs-on.dev` routes to your Vercel frontend.
   * `https://api.breastvision.runs-on.dev` routes to your PyTorch backend.

---

## 📦 5. Pre-Configured Files Summary

| File | Purpose |
| :--- | :--- |
| [`deploy/runs-on.dev/domains/breastvision.json`](./deploy/runs-on.dev/domains/breastvision.json) | The exact JSON registration file for runs-on.dev |
| [`deploy/runs-on.dev/domains/m6-cancer.json`](./deploy/runs-on.dev/domains/m6-cancer.json) | Backup subdomain record if `breastvision` is taken |
| [`deploy/Dockerfile`](./deploy/Dockerfile) | Production Dockerfile bundling FastAPI backend + Frontend |
| [`deploy/docker-compose.yml`](./deploy/docker-compose.yml) | Local & containerized testing configuration |
| [`deploy/requirements.txt`](./deploy/requirements.txt) | Lean Python dependencies for PyTorch CPU + FastAPI |
| [`deploy/render.yaml`](./deploy/render.yaml) | 1-Click Render blueprint configuration |
| [`deploy/vercel.json`](./deploy/vercel.json) | Vercel rewrite configuration for multi-service setup |

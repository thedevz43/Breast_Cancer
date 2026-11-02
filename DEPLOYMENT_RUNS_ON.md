# 🚀 Deployment Guide: Deploying BreastVision with MongoDB on runs-on.dev

This guide provides step-by-step instructions for deploying the **BreastVision Web App & Express API** with **MongoDB Database** support to **[runs-on.dev](https://runs-on.dev/)** and **Render**.

---

## 📌 1. How runs-on.dev Works

[runs-on.dev](https://runs-on.dev/) is an open-source, free developer subdomain registry operated by Advance Labs:
* Each subdomain is defined by a JSON file (`domains/<name>.json`) in the public repository [zordhalo/runs-on.dev](https://github.com/zordhalo/runs-on.dev).
* **runs-on.dev handles DNS and HTTPS wildcard routing**, while your code runs on your chosen hosting provider (such as [Render](https://render.com), [Vercel](https://vercel.com), or your own VPS).
* You can configure your root domain (e.g. `breastvision.runs-on.dev`).

---

## 🏗️ 2. Platform Architecture

The BreastVision production stack runs on a Node.js 22 container serving both the Express backend API and the static Expo Web Single Page Application (SPA), connected to a MongoDB database:

* **`https://breastvision.runs-on.dev/`** — BreastVision Mobile Ultrasound Decision Support Web App
* **`https://breastvision.runs-on.dev/api/health`** — Server Health & Service Status
* **`https://breastvision.runs-on.dev/api/database/status`** — MongoDB Connection & Collection Statistics
* **`https://breastvision.runs-on.dev/api/studies`** — Ultrasound Studies Persistence (GET/POST)
* **`https://breastvision.runs-on.dev/api/trpc`** — Type-Safe RPC Procedures for Auth, Studies, & Admin

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
                         │   Node.js 22 Container      │
                         │ ├─ / (Expo Web SPA)         │
                         │ ├─ /api/* (Express + tRPC)  │
                         │ └─ /api/database/status     │
                         └──────────────┬──────────────┘
                                        │
                                        ▼ MONGODB_URI
                         ┌─────────────────────────────┐
                         │   MongoDB Database          │
                         │ ├─ users (Auth & Roles)     │
                         │ └─ studies (Scans & Triage) │
                         └─────────────────────────────┘
```

---

## 🗄️ 3. MongoDB Database Configuration

BreastVision supports MongoDB for storing user profiles and ultrasound scan studies (including triage classifications, calibrated malignancy probabilities, image URIs, and quality control scores).

### Step 1: Create a Free MongoDB Database
You can use a free cloud MongoDB instance on **MongoDB Atlas**:
1. Go to [mongodb.com/atlas](https://www.mongodb.com/atlas) and create a free Shared Cluster (M0).
2. Create a Database User with username and password.
3. In **Network Access**, allow access from anywhere (`0.0.0.0/0`) or your Render IP.
4. Click **Connect** → **Drivers** (Node.js) and copy the connection string:
   ```
   mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/breastvision?retryWrites=true&w=majority
   ```

### Step 2: Configure Environment Variables in Render
In your **Render Dashboard** → Your Web Service → **Environment**, add:

| Key | Value | Description |
| :--- | :--- | :--- |
| `NODE_ENV` | `production` | Production mode |
| `PORT` | `3000` | Port for Express server |
| `MONGODB_URI` | `mongodb+srv://user:pass@cluster.mongodb.net/breastvision` | MongoDB connection string |
| `JWT_SECRET` | *(Random 32+ character string)* | Secret for auth tokens |
| `EXPO_PUBLIC_API_URL` | `https://breastvision.runs-on.dev` | Public URL for API calls |

*(If `MONGODB_URI` is omitted, the app will continue to run with safe in-memory fallback for offline testing).*

---

## 📋 4. Step-by-Step Deployment Instructions

### Step 1: Deploy to Render
1. Push your repository to GitHub:
   ```bash
   git push origin main
   ```
2. Log into [Render](https://dashboard.render.com).
3. If using **Docker** runtime: Render automatically picks up `deploy/Dockerfile`.
4. If using **Blueprint**: Render connects via `deploy/render.yaml`.
5. Set `MONGODB_URI` in the Environment tab.

### Step 2: Register on runs-on.dev
1. Fork the [zordhalo/runs-on.dev](https://github.com/zordhalo/runs-on.dev) repository on GitHub.
2. In your fork, create `domains/breastvision.json`:
   ```json
   {
     "name": "breastvision",
     "owner": {
       "github": "thedevz43"
     },
     "claimedAt": "2026-10-28T15:45:00.000Z",
     "records": {
       "CNAME": "breast-cancer-6el4.onrender.com"
     }
   }
   ```
3. Open a Pull Request to merge your subdomain into the main registry.

### Step 3: Verify Your Deployment
Open your browser and navigate to:
* **Web App**: `https://breastvision.runs-on.dev/`
* **Health Check**: `https://breastvision.runs-on.dev/api/health`
* **MongoDB Status**: `https://breastvision.runs-on.dev/api/database/status`
* **Studies API**: `https://breastvision.runs-on.dev/api/studies`

---

## 🐳 5. Local Docker Compose Testing with MongoDB

You can run both BreastVision and a local MongoDB instance with one command:
```bash
docker-compose -f deploy/docker-compose.yml up --build
```
This will start:
* BreastVision on `http://localhost:3000`
* MongoDB on `localhost:27017` with persistent volume `mongo-data`

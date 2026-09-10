# SafeStreet Free Cloud Deployment Guide

This guide details how to deploy SafeStreet on a **100% free domain with a secured HTTPS link** (`https://...`).

---

## Option 1: Streamlit Community Cloud (Recommended — Permanent Free Subdomain)

Streamlit Community Cloud is free forever, hosts directly from GitHub, and provides a secured HTTPS domain:  
👉 **`https://your-custom-name.streamlit.app`**

### Step 1: Push to GitHub
1. Go to [github.com/new](https://github.com/new) and create a new repository named `safestreet` (Public).
2. Open PowerShell in `E:\SIH\Project\` and run:
   ```bash
   git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/safestreet.git
   git branch -M main
   git push -u origin main
   ```

### Step 2: Deploy on Streamlit Cloud
1. Visit **[share.streamlit.io](https://share.streamlit.io)** and log in with your GitHub account.
2. Click **"New app"**.
3. Fill in the fields:
   - **Repository:** `<YOUR_GITHUB_USERNAME>/safestreet`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL (Custom Subdomain):** Choose your custom name, e.g. `safestreet-audit.streamlit.app`
4. Click **Deploy!**
5. Streamlit will install packages from `packages.txt` and `requirements.txt` and launch your app with a secure green padlock (`https://`).

---

## Option 2: Hugging Face Spaces (16 GB Free RAM + Free HTTPS)

Hugging Face Spaces provides **16 GB RAM and 2 vCPUs for free**, which is fantastic for multi-model AI (YOLO + TensorFlow).

👉 **`https://huggingface.co/spaces/<YOUR_USERNAME>/safestreet`**

### Steps:
1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Set Space name to `safestreet`.
3. Select **Streamlit** as the Space SDK.
4. Select **Public** and **Free (CPU Basic - 16GB)**.
5. Clone the space or push your existing repository:
   ```bash
   git remote add hf https://huggingface.co/spaces/<YOUR_USERNAME>/safestreet
   git push hf main
   ```
6. Your app is live with full HTTPS!

---

## Option 3: Instant 30-Second Secured Live Link (Local Machine)

If you need an immediate secured HTTPS link to share with judges or team members **right now**:

1. In Terminal 1, start the app:
   ```bash
   streamlit run app.py
   ```
2. In Terminal 2, launch a secured public tunnel using LocalTunnel (Node/npx is already installed on your system):
   ```bash
   npx localtunnel --port 8501 --subdomain safestreet-demo
   ```
3. You will immediately get a secured HTTPS link:
   👉 **`https://safestreet-demo.loca.lt`**

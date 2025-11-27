# Deploying to Railway

This app is now configured to work with browser webcam and can be deployed to Railway!

## Quick Deploy Steps:

### 1. Install Railway CLI (Optional)
```bash
npm install -g @railway/cli
railway login
```

### 2. Or Deploy via GitHub:

1. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Body Language Detector"
   git branch -M main
   git remote add origin YOUR_GITHUB_REPO_URL
   git push -u origin main
   ```

2. **Connect to Railway**:
   - Go to [railway.app](https://railway.app)
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository
   - Railway will auto-detect the configuration

3. **That's it!** Railway will:
   - Install dependencies from `requirements.txt`
   - Use the `Procfile` or `railway.toml` for deployment
   - Provide you with a public URL

## How It Works Now:

- **Browser captures webcam** using JavaScript `getUserMedia`
- **Frames sent to server** at ~10 FPS for processing
- **Server processes** with MediaPipe + ML model
- **Results streamed back** to browser with landmarks drawn
- **Works anywhere** - no server webcam needed!

## Local Testing:

```bash
python3 web_app.py
# Visit http://localhost:5001
```

## Environment Variables (if needed):

Railway automatically sets `PORT` - no configuration needed!

## Features:

- ✅ Browser webcam access
- ✅ Real-time emotion detection
- ✅ MediaPipe pose/face tracking
- ✅ Temporal smoothing for stable predictions
- ✅ Clean minimal UI
- ✅ Mobile responsive

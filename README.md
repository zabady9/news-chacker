# 🔍 Fact Checker API

AI-powered fact-checking API using BERT, DuckDuckGo search, and Google Gemini.

## 🚀 Deploy to Railway

### Prerequisites
- GitHub account
- Railway account (free)
- HuggingFace account (free)
- Google AI Studio account (free)

### 📁 Project Structure
```
fact-checker-api/
├── main.py                 # Main FastAPI application
├── requirements.txt        # Python dependencies
├── railway.json           # Railway configuration
├── nixpacks.toml          # Build configuration
├── Procfile               # Process configuration
├── .env.example           # Environment variables template
└── README.md              # This file
```

### 🔑 Environment Variables Required

1. **HUGGINGFACE_TOKEN**: Get from [HuggingFace Tokens](https://huggingface.co/settings/tokens)
2. **GOOGLE_API_KEY**: Get from [Google AI Studio](https://aistudio.google.com/app/apikey)

### 🛠️ Deployment Steps

1. **Fork/Clone this repository**
2. **Sign up at [Railway](https://railway.app)** (no credit card needed)
3. **Connect your GitHub repository**
4. **Set environment variables in Railway dashboard**
5. **Deploy!**

### 📊 API Endpoints

- `GET /` - API status and information
- `GET /health` - Health check
- `POST /warmup` - Pre-load models
- `POST /Trustness/` - Fact-check endpoint

### 🧪 Testing

```bash
# Health check
curl https://your-app.railway.app/health

# Fact check
curl -X POST https://your-app.railway.app/Trustness/ \
  -H "Content-Type: application/json" \
  -d '{"input_str": "Climate change is a hoax"}'
```

### 💰 Railway Free Tier

- $5 free credits monthly
- 512MB RAM
- 1GB storage
- No sleep policy
- Custom domains

### 🔧 Performance Tips

1. Call `/warmup` endpoint after deployment
2. Models load lazily on first request
3. Use CPU-optimized for free tier
4. Monitor usage in Railway dashboard

### 🐛 Troubleshooting

**Build fails?**
- Check requirements.txt syntax
- Verify Python version compatibility

**Out of memory?**
- Models are CPU-optimized
- Lazy loading reduces memory usage
- Consider upgrading if needed

**Slow responses?**
- Call `/warmup` to pre-load models
- First request will be slower (cold start)

### 📝 License

MIT License - feel free to use and modify!
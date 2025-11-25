# I4CU - Deepfake Detection Suite

A comprehensive deepfake detection solution with both **CLI** and **Web** interfaces.

## 📦 Repository Structure

This is a monorepo containing two complementary projects:

```
i4cu-combined/
├── i4cu-cli/          # Python CLI tool for local deepfake detection
└── i4cu-web/          # Cloudflare Workers web app for online detection
```

## 🎯 Projects Overview

### 🔧 i4cu-cli (Command Line Interface)

A powerful Python-based CLI tool for detecting deepfakes in images, videos, and audio files locally on your machine.

**Key Features:**
- ✅ Multi-format support (images, videos, audio)
- ✅ EXIF metadata analysis
- ✅ OCR text detection
- ✅ ML model integration (CLIP-ViT, UIA-ViT, Face X-Ray, etc.)
- ✅ Ensemble detection for higher accuracy
- ✅ Batch processing
- ✅ Detailed analysis reports

**Best For:**
- Local file analysis
- Batch processing
- Integration into scripts/workflows
- Offline detection
- Advanced users who want full control

**Quick Start:**
```bash
cd i4cu-cli
pip install -r requirements.txt
python cli.py image.jpg
```

📖 **[Full CLI Documentation →](i4cu-cli/README.md)**

---

### 🌐 i4cu-web (Web Application)

A modern web application hosted on Cloudflare Workers that provides AI-powered deepfake detection through your browser.

**Key Features:**
- ✅ Browser-based interface (no installation needed)
- ✅ Cloudflare Workers AI integration
- ✅ Multiple AI models (LLaVA, Llama 3.3)
- ✅ Watermark detection
- ✅ Detection history tracking
- ✅ Batch image upload
- ✅ Real-time analysis

**Best For:**
- Quick online checks
- Sharing with others
- No local installation required
- Cloud-based processing
- User-friendly interface

**Quick Start:**
```bash
cd i4cu-web
npm install
npm run dev  # Local development
npm run deploy  # Deploy to Cloudflare
```

📖 **[Full Web Documentation →](i4cu-web/README.md)**

---

## 🚀 Quick Comparison

| Feature | i4cu-cli | i4cu-web |
|---------|----------|----------|
| **Platform** | Local (Python) | Cloud (Cloudflare Workers) |
| **Installation** | Required | None (browser-based) |
| **File Types** | Images, Videos, Audio | Images |
| **ML Models** | PyTorch models (CLIP-ViT, etc.) | Cloudflare AI (LLaVA, Llama) |
| **Processing** | Your machine | Cloudflare edge |
| **Batch Support** | ✅ Full batch processing | ✅ Multiple images |
| **Offline** | ✅ Works offline | ❌ Requires internet |
| **Cost** | Free (local) | Free tier available |
| **Best For** | Power users, automation | Quick checks, sharing |

## 📋 Requirements

### For i4cu-cli:
- Python 3.8+
- pip
- Tesseract OCR (for OCR analysis)
- Optional: PyTorch (for ML models)

### For i4cu-web:
- Node.js 18+
- npm
- Cloudflare account (for deployment)
- Cloudflare Workers AI enabled

## 🛠️ Installation

### Install CLI Tool
```bash
cd i4cu-cli
pip install -r requirements.txt
```

### Setup Web App
```bash
cd i4cu-web
npm install
npx wrangler login
```

## 📚 Documentation

### CLI Documentation
- **[Main README](i4cu-cli/README.md)** - Complete CLI guide
- **[Quick Start](i4cu-cli/QUICKSTART.md)** - Get started in 5 minutes
- **[Model Setup](i4cu-cli/MODEL_SETUP.md)** - Setting up ML models
- **[Ensemble Guide](i4cu-cli/ENSEMBLE_GUIDE.md)** - Using ensemble detection
- **[Scoring Explained](i4cu-cli/HOW_SCORING_WORKS.md)** - How detection works

### Web Documentation
- **[Web README](i4cu-web/README.md)** - Complete web app guide

## 🎯 Use Cases

### When to Use i4cu-cli:
- ✅ Analyzing large batches of files
- ✅ Processing videos and audio
- ✅ Integration into automated workflows
- ✅ Working with sensitive data (stays local)
- ✅ Need for specific ML models
- ✅ Offline detection

### When to Use i4cu-web:
- ✅ Quick one-off image checks
- ✅ Sharing detection with others
- ✅ No installation required
- ✅ Mobile/tablet access
- ✅ Cloud-based processing

## 🔄 Can I Use Both?

**Absolutely!** They work independently:
- Use **i4cu-cli** for local, batch processing, and advanced use cases
- Use **i4cu-web** for quick online checks and sharing

They don't need to be connected - each serves its own purpose.

## 📊 Performance Metrics

The suite has been rigorously tested to ensure reliability across various media types. In initial benchmarking on a dataset of **300+ media files**:

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **75-80%** |
| **Precision** | **75%** |
| **Recall** | **83%** |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests for either project.

## 📄 License

This project is provided as-is for educational and research purposes.

## 🔗 Links

- **CLI Tool**: [i4cu-cli/README.md](i4cu-cli/README.md)
- **Web App**: [i4cu-web/README.md](i4cu-web/README.md)

---

**Note**: Both tools are independent and can be used separately. Choose the one that best fits your needs!


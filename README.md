# 🦄 Kickstarter Counterfactual Simulator: The Ultimate Edition

<!-- Header Images -->
<p align="center">
  <img src="assets/hero_banner.png" width="32%" alt="Hero Banner">
  <img src="assets/deep_analysis.png" width="32%" alt="Deep Analysis">
  <img src="assets/ai_brain.png" width="32%" alt="AI Brain">
</p>

> **"Where Deep Learning Meets Causal Economics"**
> *An autonomous system that predicts success, optimizes pricing through causal inference, and provides strategic consulting via AI.*

---

## 🆕 What's New (Latest Update)

| Feature                 | Description                               |
| ----------------------- | ----------------------------------------- |
| 🌍 **Multi-Platform**    | Now supports Kickstarter + Indiegogo data |
| 📊 **8+ Visualizations** | Charts, histograms, scatter plots, radar  |
| ☁️ **Cloud Ready**       | Deploy to Streamlit Cloud in 2 minutes    |
| 🔄 **Auto-Updates**      | Scheduled data refresh via cron           |
| 🎨 **React Frontend**    | Optional Lovable.dev-generated UI         |

---

## 💡 What Is This Project?

Imagine you are launching a product on Kickstarter. You have two burning questions:
1.  **"Will I succeed?"** (Prediction)
2.  **"What price should I charge?"** (Strategy)

### 🚫 The Old Way (Standard AI)
A normal AI might say: *"Projects with high prices fail."*
So you lower your price... and **fail anyway**. Why? Because it confused *correlation* with *causation*.

### ✅ The New Way (This Project)
This system uses **Causal Inference** to simulate: *"If we took YOUR project and increased the price by $10, what happens?"*

---

## 📊 Dashboard Visualizations (8+)

| Chart                           | What It Shows                        |
| ------------------------------- | ------------------------------------ |
| **Counterfactual Demand Curve** | Price vs Funding Ratio               |
| **Goal Sensitivity**            | How goal affects success probability |
| **Category Benchmark Radar**    | You vs Market Average                |
| **Feature Importance**          | What drives the AI's predictions     |
| **Similar Campaigns Bar**       | Your prediction vs similar projects  |
| **Category Success Rates**      | Bar chart of success by category     |
| **Funding Distribution**        | Histogram of all funding ratios      |
| **Duration vs Success Scatter** | Campaign length impact               |
| **Goal vs Backers Correlation** | Scatter with color gradient          |

---

## 🚀 Quick Start

### Option 1: Local Development
```bash
# Clone & Install
git clone https://github.com/imshivanshutiwari/kickstarter-causal-ai.git
cd kickstarter-causal-ai
pip install -r requirements.txt

# Run the Manager
python manage_data.py
# Select Option 5 → Train & Launch Dashboard
```

### Option 2: Cloud Deployment (FREE)
1. Go to **https://share.streamlit.io**
2. Connect your GitHub
3. Select repo: `imshivanshutiwari/kickstarter-causal-ai`
4. Main file: `app.py`
5. Click **Deploy!**

---

## 🎛️ CLI Manager Options

```bash
python manage_data.py
```

| Option | Action                                         |
| ------ | ---------------------------------------------- |
| 1      | Smart Update (Kaggle) - Check for new data     |
| 2      | Force Full Update - Re-download everything     |
| 3      | Scrape Kickstarter Live - Selenium scraper     |
| 4      | Scrape Indiegogo - Multi-platform support      |
| 5      | Run Pipeline - Train models & launch dashboard |
| 6      | Exit                                           |

---

## 🧮 Mathematical Foundation

We use **Two-Stage Least Squares (2SLS)** with BERT embeddings:

**Stage 1:** Predict "Clean Price" using instruments:
$$ \hat{P} = \alpha_0 + \alpha_1 Z + \alpha_2 X_{nlp} + \epsilon_1 $$

**Stage 2:** Estimate causal effect:
$$ Y = \beta_0 + \beta_{price} \hat{P} + \beta_2 X_{nlp} + \epsilon_2 $$

Where $\beta_{price}$ is the **True Causal Effect** of price on success.

---

## 🌲 Directory Structure

```text
kickstarter-causal-ai/
├── data/
│   ├── raw/                 # Kaggle + Scraped data
│   └── processed/           # Feature-engineered files
├── src/
│   ├── ai_consultant.py     # 🤖 AI Strategy Agent
│   ├── nlp_features.py      # 🧠 BERT Embeddings
│   ├── live_scraper.py      # 🕷️ Kickstarter Scraper
│   ├── indiegogo_scraper.py # 🕷️ Indiegogo Scraper (NEW!)
│   ├── train_models.py      # 📉 Causal Models
│   ├── api.py               # 🚀 FastAPI Backend
│   └── scheduled_update.py  # ⏰ Cron Job Handler
├── manage_data.py           # 🎛️ Unified CLI
├── run_pipeline.py          # ⚙️ Orchestrator
├── run_fullstack.py         # 🌐 API + React Launcher
├── app.py                   # 📊 Streamlit Dashboard
├── Dockerfile               # 🐳 Container Ready
└── requirements.txt         # 📦 Dependencies
```

---

## 🤖 AI Components

| Component         | Model                  | Purpose                      |
| ----------------- | ---------------------- | ---------------------------- |
| **NLP Engine**    | MiniLM-L6-v2           | Controls for project quality |
| **Causal Models** | 2SLS + Causal Forest   | Price elasticity estimation  |
| **AI Consultant** | Rule-based + LLM-ready | Strategic recommendations    |

---

## 🔄 Automatic Data Updates

For production, set up a cron job:

```bash
# Run daily at 3 AM
0 3 * * * cd /path/to/project && python src/scheduled_update.py
```

---

## 🛡️ Robustness Certification

| Test             | Status   |
| ---------------- | -------- |
| Unit Tests       | ✅ PASSED |
| API Stress       | ✅ PASSED |
| Data Safety      | ✅ PASSED |
| Math Validation  | ✅ PASSED |
| Cloud Deployment | ✅ READY  |

---

## 📱 Alternative: React Frontend

A premium React + Tailwind frontend is available:

```bash
cd "page design"
npm install
npm run dev
# Opens at http://localhost:8080

# Run API separately:
python -m uvicorn src.api:app --reload --port 8000
```

---

## ❓ FAQ

**Q: Chromedriver error?**
> Install Google Chrome. Driver auto-downloads.

**Q: NLP is slow?**
> CPU needs ~1 min. GPU is seconds.

**Q: Why is price effect positive?**
> Your category is a "Premium/Veblen Good" - higher prices signal quality!

---

> *Built with Causal AI by Antigravity Agent* 🦄

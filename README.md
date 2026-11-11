# 🤖 AutoML Agent for Tabular Data

> A multi-agent LLM system that automates the full machine learning pipeline — from CSV upload to model selection and AI-generated reports — served through a premium Streamlit interface.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green)
![Gemini](https://img.shields.io/badge/Google%20Gemini-API-blue?logo=google)

---

## ✨ Features

| Feature | Details |
|---|---|
| **Data Profiling** | Missing values, distributions, skewness, outliers, correlations |
| **Auto Task Detection** | Automatically classifies task as Classification or Regression |
| **Feature Engineering** | Imputation, StandardScaler, OneHotEncoder via sklearn pipelines |
| **LLM Suggestions** | Gemini-powered feature engineering insights |
| **Model Training** | 6–7 models, 5-fold cross-validation, ranked leaderboard |
| **Models Supported** | Logistic Regression, Random Forest, Gradient Boosting, XGBoost, SVM, KNN, Linear/Ridge Regression, SVR |
| **AI Report** | Gemini-generated narrative covering data quality, model selection, key insights |
| **PDF Export** | Downloadable PDF report with charts, leaderboard, and AI analysis |
| **Dark UI** | Premium dark-themed Streamlit interface with glassmorphism effects |

---

## 🏗 Architecture

```
User uploads CSV
       │
       ▼
┌─────────────────┐
│  Orchestrator   │  ← Coordinates the pipeline
└────────┬────────┘
         │
   ┌─────┴──────┬──────────────┬────────────────┐
   ▼            ▼              ▼                ▼
Data          Feature       Model            Report
Profiler      Engineer      Trainer          Generator
Agent         Agent         Agent            Agent
```

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/automl-agent-tabular.git
cd automl-agent-tabular
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate    # macOS/Linux
# venv\Scripts\activate     # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API key (optional but recommended)
```bash
cp .env.example .env
# Edit .env and add your Gemini API key
# Get a free key at https://aistudio.google.com/
```

### 5. Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📁 Project Structure

```
automl-agent-tabular/
├── app.py                      # Streamlit entrypoint
├── requirements.txt
├── .env.example
├── README.md
│
├── agents/
│   ├── orchestrator.py         # Pipeline coordinator
│   ├── data_profiler.py        # Agent 1 — Data profiling
│   ├── feature_engineer.py     # Agent 2 — Feature preparation
│   ├── model_trainer.py        # Agent 3 — Training + CV
│   └── report_generator.py     # Agent 4 — AI reports
│
├── utils/
│   ├── llm_client.py           # Gemini API wrapper + fallback
│   ├── visualizations.py       # Chart generation
│   └── pdf_export.py           # PDF report builder
│
└── sample_data/
    ├── iris.csv
    └── california_housing.csv
```

---

## 🔧 Usage

1. **Upload a CSV** — drag & drop in the sidebar
2. **Configure settings** — test size, CV folds, target column
3. **Click "Run AutoML Pipeline"**
4. **Explore results** across 4 tabs:
   - 📊 **Data Profile** — statistics, charts, correlations
   - 🛠 **Feature Engineering** — steps applied + LLM suggestions
   - 🏆 **Model Leaderboard** — ranked models with CV scores
   - 📄 **AI Report** — narrative analysis + PDF download

---

## 🤖 LLM Integration

The system uses **Google Gemini 1.5 Flash** for:
- Feature engineering suggestions based on column semantics
- Professional narrative report generation
- Data quality assessment

**Without an API key**, the app still functions fully — the ML pipeline (profiling, training, leaderboard) works normally, and LLM outputs are replaced with rule-based fallbacks.

---

## 📊 Supported Models

### Classification
| Model | Library |
|---|---|
| Logistic Regression | scikit-learn |
| Random Forest | scikit-learn |
| Gradient Boosting | scikit-learn |
| XGBoost | xgboost |
| SVM (RBF kernel) | scikit-learn |
| K-Nearest Neighbors | scikit-learn |

### Regression
| Model | Library |
|---|---|
| Linear Regression | scikit-learn |
| Ridge Regression | scikit-learn |
| Random Forest | scikit-learn |
| Gradient Boosting | scikit-learn |
| XGBoost | xgboost |
| SVR (RBF kernel) | scikit-learn |
| K-Nearest Neighbors | scikit-learn |

---

## 🛠 Tech Stack

- **UI**: Streamlit 1.35+
- **ML**: scikit-learn 1.3+, XGBoost 2.0+
- **LLM**: Google Gemini 1.5 Flash
- **Data**: pandas, numpy, scipy
- **Visualization**: matplotlib, seaborn
- **PDF**: ReportLab
- **Config**: python-dotenv

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

## 👤 Author

Built by **Neeraj Bolla** — Nov–Dec 2025

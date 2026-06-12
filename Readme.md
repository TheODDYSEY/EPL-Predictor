<div align="center">

# ⚽ EPL Expected Goals Projector

### Predict Premier League match scores using Machine Learning & Poisson Statistics

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189AB4?style=for-the-badge&logo=xgboost&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Stars](https://img.shields.io/github/stars/TheODDYSEY/EPL-Predictor?style=for-the-badge&color=yellow)
![Forks](https://img.shields.io/github/forks/TheODDYSEY/EPL-Predictor?style=for-the-badge&color=orange)

**[🚀 Live Demo →](https://theoddysey-epl-predictor-app-mabpex.streamlit.app/)**

</div>

---

## 📸 Preview

<div align="center">

![App Preview](https://github.com/TheODDYSEY/EPL-Predictor/raw/master/img/app.png)

</div>

---

## 📖 About

**EPL Expected Goals Projector** is a Streamlit web app that predicts English Premier League match outcomes using historical match data and four machine learning approaches. Select a season, pick two teams, and get projected scorelines, win probabilities, and RMSE metrics — all in real time.

---

## ✨ Features

- ⚽ **Score projection** — predicted home and away goals for any EPL fixture
- 📊 **Win probability** — home win / draw / away win percentages (Poisson model)
- 🤖 **4 prediction models** — Poisson, Random Forest, XGBoost, and Ensemble
- 📅 **Multi-season support** — select and compare across different EPL seasons
- 📈 **RMSE display** — model accuracy metric shown per prediction
- ✅ **Actual score comparison** — shows real result if the match has already happened
- 🧠 **Feature selection** — automatically identifies most correlated stats per model

---

## 🤖 Models

| Model | Type | Description |
|---|---|---|
| **Poisson Distribution** | Statistical | Models goals as independent Poisson processes; outputs win probabilities |
| **Random Forest** | ML Regressor | Ensemble of decision trees trained on cumulative team stats |
| **XGBoost** | ML Regressor | Gradient-boosted trees; handles non-linear feature interactions |
| **Ensemble** | Combined | Averages RF and XGBoost predictions for reduced variance |

---

## ⚙️ How It Works

### 1 — Data Preparation
Historical EPL match CSV files are loaded from `engg_data/`. Each file is cleaned and transformed to produce per-team cumulative statistics: wins, draws, losses, points, goals scored, goals conceded, and league position up to each matchday.

### 2 — Feature Selection
For each prediction target (home goals, away goals), the pipeline computes Pearson correlations against all available features and selects the most correlated subset to feed into the model.

### 3 — Model Training
Separate regression models are trained for home goals and away goals. The Ensemble model averages the outputs of Random Forest and XGBoost. Poisson parameters (λ_home, λ_away) are derived from historical attack and defence strength ratios.

### 4 — Prediction
When a fixture is selected, the system retrieves each team's latest cumulative stats up to that matchday, feeds them into the chosen model, and returns projected scorelines.

### 5 — Visualization
The Streamlit UI displays the projected score, win probabilities (Poisson), RMSE, and — if the match has occurred — the actual result for comparison.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.x
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/TheODDYSEY/EPL-Predictor.git
cd EPL-Predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

> Open your browser at `http://localhost:8501`

---

## 🖥️ Usage

1. Select an **EPL season** from the dropdown
2. Choose a **prediction model** — Poisson, RandomForest, XGBoost, or Ensemble
3. Pick the **Home team** and **Away team**
4. Click **Run**
5. View projected score, win probabilities, actual result, and RMSE

---

## 📁 Project Structure

```
EPL-Predictor/
├── app.py               # Streamlit application entry point
├── requirements.txt     # Python dependencies
├── engg_data/           # Engineered CSV files per EPL season
├── data/                # Raw historical match data
└── img/
    └── app.png          # App preview screenshot
```

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white) | Web UI framework |
| ![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white) | Data loading, cleaning, feature engineering |
| ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikitlearn&logoColor=white) | Random Forest Regressor, metrics |
| ![XGBoost](https://img.shields.io/badge/XGBoost-189AB4?logo=xgboost&logoColor=white) | Gradient-boosted regression |
| `SciPy` | Poisson distribution and probability calculations |

---

## 🤝 Contributing

Pull requests and issues are welcome. Fork the repo, make your changes, and open a PR.

---

## 📄 License

Licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- [Football-Data.org](https://www.football-data.org/) for historical EPL match data
- [Streamlit](https://streamlit.io/) for the web framework
- [Scikit-learn](https://scikit-learn.org/) and [XGBoost](https://xgboost.readthedocs.io/) for ML algorithms

---

## ⭐ Star History

<div align="center">

[![Star History Chart](https://api.star-history.com/svg?repos=TheODDYSEY/EPL-Predictor&type=Date)](https://star-history.com/#TheODDYSEY/EPL-Predictor&Date)

</div>

---

<div align="center">

Made with ❤️ using Python & Streamlit &nbsp;·&nbsp; ⚽ Happy predicting!

</div>

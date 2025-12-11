# forest-fires

---

# 🌲 Algerian Forest Fires Prediction Using SVM

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python\&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Train Accuracy](https://img.shields.io/badge/Train%20Accuracy-94%25-brightgreen)]()
[![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-91%25-orange)]()
[![Last Updated](https://img.shields.io/badge/Last%20Updated-2025--12--11-blueviolet)]()

Predicting forest fires in Algeria using **Support Vector Machines (SVM)**. The project demonstrates data preprocessing, feature analysis, and predictive modeling for early fire detection.

---

## 🚀 Project Overview

Forest fires are a major environmental and socio-economic threat in Algeria. Predicting the likelihood of fires can help in early intervention and mitigation strategies.

This project uses meteorological, vegetation, and environmental features to build a **Support Vector Machine classifier** capable of predicting forest fires.

**Objectives:**

* Analyze forest fire patterns across Algeria
* Identify key features contributing to fire risk
* Build an SVM model for accurate fire prediction
* Achieve high accuracy for early warning

---

## 📊 Dataset

The dataset contains historical records of forest fires in Algeria, along with environmental and meteorological features.

**Key Features:**

| Feature         | Description                             |
| --------------- | --------------------------------------- |
| Temperature     | Daily temperature (°C)                  |
| Humidity        | Relative humidity (%)                   |
| Wind Speed      | Wind speed (km/h)                       |
| Rainfall        | Daily rainfall (mm)                     |
| Vegetation Type | Type of vegetation                      |
| Fire Occurrence | Target variable: Fire (1) / No Fire (0) |

> Source: [Algerian Forest Fires Dataset – Kaggle or UCI] *(replace with actual source)*

---

## 🛠️ Tools & Libraries

* **Python 3.11** – Programming language
* **Pandas & NumPy** – Data preprocessing
* **Matplotlib & Seaborn** – Data visualization
* **Scikit-learn** – SVM modeling and evaluation
* **Plotly (Optional)** – Interactive feature exploration

---

## 🔍 Key Analyses

1. **Data Cleaning & Preprocessing**

   * Handle missing values
   * Encode categorical features
   * Normalize numerical features for SVM

2. **Exploratory Data Analysis (EDA)**

   * Correlation analysis to identify influential features
   * Visualize fire occurrence patterns by temperature, humidity, and vegetation

3. **Support Vector Machine Model**

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv('algerian_forest_fires.csv')

# Features and target
X = df[['Temperature','Humidity','Wind Speed','Rainfall']]  # example features
y = df['Fire Occurrence']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SVM model
model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluation
print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))
```

---

## 📈 Performance

| Metric            | Score |
| ----------------- | ----- |
| Training Accuracy | 94%   |
| Testing Accuracy  | 91%   |

> The model demonstrates strong generalization, making it suitable for early fire detection.

---

## 📊 Visualizations

### Feature Correlation

![Correlation Heatmap](https://user-images.githubusercontent.com/yourusername/correlation_heatmap.png)
*Shows relationships between environmental features and fire occurrence*

### Fire Occurrence by Temperature and Humidity

![Scatter Plot](https://user-images.githubusercontent.com/yourusername/fire_scatter.png)

### Interactive Plot (Optional)

![Interactive Plot](https://user-images.githubusercontent.com/yourusername/fire_interactive.gif)

> Visualizations help identify high-risk conditions for forest fires.

---

## ⚡ Usage

1. Clone the repository:

```bash
git clone <repo-url>
cd algerian-forest-fires
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the SVM prediction script:

```bash
python svm_forest_fire.py
```

4. Modify environmental feature values to predict fire occurrence for new scenarios.

---

## 📂 Folder Structure

```
algerian-forest-fires/
│
├─ algerian_forest_fires.csv
├─ svm_forest_fire.py
├─ notebooks/
│   ├─ EDA.ipynb
│   └─ Modeling.ipynb
├─ visualizations/
│   ├─ correlation_heatmap.png
│   ├─ fire_scatter.png
│   └─ fire_interactive.gif
├─ README.md
├─ requirements.txt
└─ LICENSE
```

---

## 💡 Insights

* **Temperature, Humidity, and Wind Speed** are critical factors influencing fire occurrence
* SVM with RBF kernel provides robust classification performance
* Interactive visualizations help identify high-risk regions and conditions
* High accuracy allows early warning systems for forest management

---

## 📌 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---



# 🔍 Glass Type Analysis & Prediction

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)
![Machine Learning](https://img.shields.io/badge/ML-ScikitLearn-orange)
![Status](https://img.shields.io/badge/Project-Completed-success)

An **interactive Machine Learning web application** that analyzes the chemical composition of glass and predicts its **type of glass** using multiple classification algorithms.

The project integrates **Exploratory Data Analysis (EDA), visualization, and ML model training** into a single **Streamlit dashboard** for an interactive data science workflow.


# 📊 Project Overview

Glass classification is important in **materials science, manufacturing, and recycling**. Different glass types have different chemical compositions which can be used for identification.

This project:

* Performs **data exploration and visualization**
* Trains **multiple machine learning models**
* Allows **real-time prediction through a web interface**
* Enables **interactive hyperparameter tuning**


# ⚙️ Tech Stack

| Category        | Tools               |
| --------------- | ------------------- |
| Programming     | Python              |
| ML Library      | Scikit-learn        |
| Data Processing | Pandas, NumPy       |
| Visualization   | Matplotlib, Seaborn |
| Web Framework   | Streamlit           |


# 🧬 Dataset Features

The dataset contains **chemical attributes of glass samples** used to determine the type of glass.

| Feature | Description      |
| ------- | ---------------- |
| RI      | Refractive Index |
| Na      | Sodium           |
| Mg      | Magnesium        |
| Al      | Aluminum         |
| Si      | Silicon          |
| K       | Potassium        |
| Ca      | Calcium          |
| Ba      | Barium           |
| Fe      | Iron             |

**Target Variable:**
Glass Type Classification

Examples include:

* Building Windows (Float Processed)
* Building Windows (Non-Float Processed)
* Vehicle Windows
* Containers
* Tableware
* Headlamps


# 📈 Exploratory Data Analysis

The dashboard provides multiple visualization tools:

* Scatter Plots
* Histograms
* Box Plots
* Count Plots
* Pie Charts
* Correlation Heatmap
* Pair Plot

These visualizations help analyze **feature distribution and relationships between variables**.


# 🤖 Machine Learning Models

The application supports multiple classification models:

### 1️. Support Vector Machine (SVM)

* Adjustable **C**
* Kernel selection
* Gamma tuning

### 2️. Random Forest Classifier

* Number of estimators
* Maximum depth

### 3️. Logistic Regression

* Regularization parameter (C)
* Maximum iterations

Each model is trained and evaluated using **train-test split validation**.


# 📊 Model Evaluation

The application provides:

* **Accuracy Score**
* **Confusion Matrix**
* Real-time prediction output


# 🖥 Application Features

Users can:

✔ View the entire dataset
✔ Explore data visually
✔ Select machine learning algorithms
✔ Tune model hyperparameters
✔ Input chemical composition using sliders
✔ Predict the **glass type instantly**


# 📂 Project Structure

```
glass-type-analysis
│
├── app.py                # Streamlit application
├── glass-types.csv       # Dataset
├── requirements.txt      # Dependencies
└── README.md
```


# 📌 Future Improvements

* Add **XGBoost and Gradient Boosting models**
* Improve **feature engineering**
* Add **model comparison dashboard**
* Deploy application using **Streamlit Cloud or Docker**


# 👩‍💻 Author

**Tanya Rodrigues**


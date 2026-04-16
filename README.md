# Titanic Survival Prediction: An End-to-End Machine Learning Pipeline

This repository contains a complete data science workflow predicting passenger survival on the Titanic. It demonstrates the process of transforming raw, incomplete historical data into a clean, mathematical matrix to train and evaluate binary classification models.

### 🛠 Tech Stack & Tools
* **Language:** Python 3
* **Environment:** Jupyter Notebook
* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn

### ⚙️ Core Pipeline Features
* **Exploratory Data Analysis (EDA):** Statistical pattern recognition and visual correlation analysis to identify survival trends based on socioeconomic class, sex, and age.
* **Intelligent Imputation:** Handled missing data points without skewing statistics (e.g., using regular expressions to extract passenger titles like "Master" and "Miss" to accurately calculate and impute missing ages).
* **Feature Engineering:** Combined fragmented data points into stronger predictive vectors (creating a `FamilySize` feature) and applied One-Hot Encoding to categorical variables while avoiding the dummy variable trap.
* **Predictive Modeling:** Implemented strict Train/Test splits to prevent model overfitting, establishing baselines for binary classification algorithms.

### 📂 Project Structure
* `/data` - Contains the raw `titanic.csv` and the processed `titanic_clean.csv`
* `01_eda.ipynb` - Visualizing distributions, correlations, and target variables
* `02_data_cleaning.ipynb` - Matrix preparation, feature engineering, and categorical encoding
* `03_model_building.ipynb` - Training, tuning, and evaluating predictive models
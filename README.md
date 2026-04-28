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
* **Algorithmic Stability Testing:** Executed an Anti-Luck Protocol using **5-Fold Cross-Validation** across 7 different classification architectures to verify mathematical stability over pure random chance.
* **Hyperparameter Optimization:** Deployed a multi-threaded **GridSearchCV** to brute-force hundreds of architectural combinations, preventing overfitting and finding the absolute mathematical ceiling of the top-tier engines.
* **Shadow Deployment (A/B Testing):** Engineered a dynamic data router to test new, unseen data simultaneously through both distance-based algorithms (Scaled SVM) and logic-based algorithms (Random Forest) to verify production confidence.

### 📊 Key Results & Insights
* **Top Performing Model:** The tuned **Support Vector Machine (SVM)** secured the highest accuracy with an F1-Score of **0.7360**, demonstrating ironclad stability (variance of just +/- 0.0096) during cross-validation.
* **Feature Importance:** Extracted the internal Information Gain ledger from the Random Forest, mathematically proving that `Sex`, `Fare`, and `Pclass` carried nearly 70% of the predictive weight.

### 📂 Project Structure
* `/data` - Contains the raw `titanic.csv` and the processed `titanic_clean.csv`
* `01_eda.ipynb` - Visualizing distributions, correlations, and target variables
* `02_data_cleaning.ipynb` - Matrix preparation, feature engineering, and categorical encoding
* `03_model_building.ipynb` - Algorithm compilation, cross-validation, GridSearch tuning, and custom passenger deployment.

### How to Run
To execute the pipeline locally, clone this repository and install the required dependencies:

```bash
pip install -r requirements.txt
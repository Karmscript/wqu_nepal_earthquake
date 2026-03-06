# Earthquake Damage Prediction in Nepal (Capstone Project)

This repository contains my **capstone project** from the Applied Data Science Lab at **WorldQuant University**, where I applied the **end-to-end data science workflow** to predict building damage severity during the 2015 Nepal earthquake.

##  Project Overview
The goal of this project was to build the most effective model that predicts **building damage severity** based on structural, demographic, and geographic features.  
The analysis focused on **Kavrepalanchok district** in Nepal.

Key tasks included:
- SQL queries for data wrangling and feature extraction.
- Data cleaning, preprocessing, and exploratory data analysis (EDA).
- Feature engineering and handling class imbalance.
- Building, training, and tuning **Logistic Regression** and **Decision Tree models**.
- Evaluating models using validation curves and test predictions.
- Extracting **feature importance** to understand key drivers of damage.

---

##  Workflow
1. **Data Wrangling**  
   - Queried relational tables with SQL (`id_map`, `building_structure`, `building_damage`).  
   - Created the `severe_damage` target variable (damage grade > 3 encoded as 1).  
   - Removed leakage/multicollinearity features.

2. **Exploratory Data Analysis (EDA)**  
   - Checked **class balance** of severe damage cases.  
   - Explored **plinth area vs. damage** with boxplots.  
   - Analyzed **roof type vs. damage** using pivot tables.  

3. **Modeling**  
   - Established **baseline accuracy**.  
   - Trained Logistic Regression and Decision Tree classifiers.  
   - Performed **hyperparameter tuning** (Decision Tree `max_depth`, 1–15).  
   - Selected the best model using validation accuracy.

4. **Evaluation**  
   - Tested final Decision Tree on held-out dataset.  
   - Extracted **feature importance** using Gini index.  

---

##  Key Results
- **Best model:** Decision Tree with optimal `max_depth` of 10.  
- Achieved strong performance on test data, outperforming baseline.  
- Most important features for damage prediction included:  
  - Roof type  
  - Plinth area (sq. ft.)  
  - Structural material  

---

##  Tools & Libraries
- **Python**: pandas, numpy, seaborn, matplotlib, scikit-learn, statsmodels  
- **SQL** for database queries  
- **Plotly** for interactive visualizations  

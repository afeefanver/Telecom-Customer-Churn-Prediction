# 📊 Telecom Customer Churn Prediction System

This project predicts whether a telecom customer is likely to **stay or churn** using machine learning models.  
It combines **data analysis, model building, and deployment** into a complete, production-ready workflow.

---

## 🚀 Project Overview
This end-to-end ML project helps telecom businesses identify customers who are most likely to churn, enabling proactive retention strategies.

The project includes:
- **Data preprocessing & EDA** to understand churn patterns.
- **Model training & comparison** using Logistic Regression, Random Forest, SVM, AdaBoost, and XGBoost.
- **Final model selection:** SVM (Support Vector Machine) — offering the best accuracy and recall balance.
- **Model deployment:** A Streamlit web app for real-time churn prediction.

---

## 🧩 Project Structure
| File | Description |
|------|--------------|
| `Telecom-Customer-Churn-Prediction.ipynb` | Full notebook with data analysis, visualizations, and ML model training. |
| `churn_main_pr.py` | Python script that trains the final SVM model and saves it as a `.joblib` file. |
| `svm_model.joblib` | Trained SVM model used by the Streamlit app for predictions. |
| `churn_app_pr.py` | Streamlit web app for real-time churn prediction. |

---

## 🧠 Key Insights
- **Contract Type, Tenure, and Monthly Charges** are the strongest churn indicators.
- **SVM model** performed best among all tested algorithms.
- **Streamlit app** makes predictions easy to use and visualize in real time.

---

## 🧰 Tech Stack
- Python, Pandas, NumPy, Scikit-learn  
- Streamlit, Joblib  
- Matplotlib, Seaborn  
- XGBoost, AdaBoost, Random Forest, SVM

---

## ⚙️ How to Run Locally
```bash
# Clone the repository
git clone https://github.com/yourusername/Telecom-Customer-Churn-Prediction.git

# Navigate to the folder
cd Telecom-Customer-Churn-Prediction

# Run the Streamlit app
streamlit run churn_app_pr.py

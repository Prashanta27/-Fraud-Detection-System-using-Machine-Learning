# -Fraud-Detection-System-using-Machine-Learning
This project is an end-to-end machine learning solution designed to detect fraudulent financial transactions. It includes data preprocessing, feature engineering, model training, evaluation, and deployment using a web interface.

📊 Project Overview

Financial fraud is a major problem in digital transactions. The goal of this project is to build a machine learning model that can accurately identify fraudulent transactions based on transaction patterns and account balances.

🧠 Machine Learning Workflow
 The project follows a complete ML pipeline:

1.Data Cleaning and Exploration
2Feature Engineering
3.Data Preprocessing with ColumnTransformer
4.Model Training using Logistic Regression
5.Model Evaluation (Precision, Recall, F1-score)
6.Deployment using Streamlit

📁 Dataset Features
Key features used in the model:

type – Transaction type (PAYMENT, TRANSFER, CASH_OUT)
amount – Transaction amount
oldbalanceOrg – Sender’s balance before transaction
newbalanceOrig – Sender’s balance after transaction
oldbalanceDest – Receiver’s balance before transaction
newbalanceDest – Receiver’s balance after transaction

Engineered Features:

BalanceDiffOrig
BalanceDiffDest

⚙️ Technologies Used
Python
Pandas & NumPy
Scikit-learn
Matplotlib & Seaborn
Streamlit
Pickle (for model serialization)

🏗️ Model Pipeline

The model is built using a Scikit-learn Pipeline that includes:

StandardScaler for numerical features
OneHotEncoder for categorical features
Logistic Regression with class balancing

This ensures consistent preprocessing during both training and prediction.

📈 Model Performance

The model was evaluated using:

Confusion Matrix
Classification Report
Accuracy Score: 93.88917675706435
Special focus was given to recall for fraudulent transactions, since missing a fraud case is more costly than a false positive.

🖥️ Web Application

A Streamlit web app was created to allow users to input transaction details and receive real-time fraud predictions.
To run the app locally:
streamlit run Fraud_Detection_Web.py

🚀 Future Improvements
Try advanced models such as Random Forest or XGBoost
Perform hyperparameter tuning
Deploy the app to cloud platforms like Streamlit Cloud or Render

👤 Author

Prashanta Das
Aspiring Data Scientist & Machine Learning Enthusiast

If you found this project useful, feel free to ⭐ the repository.

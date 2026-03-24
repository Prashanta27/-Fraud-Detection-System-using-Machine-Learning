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

1.type – Transaction type (PAYMENT, TRANSFER, CASH_OUT)
2.amount – Transaction amount
3.oldbalanceOrg – Sender’s balance before transaction
4.newbalanceOrig – Sender’s balance after transaction
5.oldbalanceDest – Receiver’s balance before transaction
6.newbalanceDest – Receiver’s balance after transaction

Engineered Features:

1.BalanceDiffOrig
2.BalanceDiffDest

⚙️ Technologies Used
1.Python
2.Pandas & NumPy
3.Scikit-learn
4.Matplotlib & Seaborn
5.Streamlit
6.Pickle (for model serialization)

🏗️ Model Pipeline

The model is built using a Scikit-learn Pipeline that includes:

1.StandardScaler for numerical features
2.OneHotEncoder for categorical features
3.Logistic Regression with class balancing

This ensures consistent preprocessing during both training and prediction.

📈 Model Performance

The model was evaluated using:

1.Confusion Matrix
2.Classification Report
3.Accuracy Score: 93.88917675706435
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

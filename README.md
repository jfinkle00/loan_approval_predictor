Author: Jason Finkle

Contact Information: jfinkle00@gmail.com

🏦 Loan Approval & Delinquency Prediction Using Machine Learning

📌 Overview
This project predicts whether a loan application will be approved or not based on customer financial data and loan characteristics. Using machine learning classification models, this project demonstrates feature engineering, model comparison, and performance evaluation techniques commonly used in data science.

📂 Dataset Description
Rows: 24,000 loan applicants

Columns: 7 features

Income: Applicant's annual income

Credit_Score: Credit score (300-850)

Loan_Amount: Amount requested for loan

DTI_Ratio: Debt-to-Income ratio

Employment_Status: Employed or Unemployed

Text: Loan purpose description

Approval: Loan status (Approved/Rejected)

✅ No missing values
✅ Cleaned categorical variables using One-Hot Encoding
✅ Normalized continuous features using MinMaxScaler

🧹 Data Preprocessing
Dropped irrelevant text descriptions

Converted categorical variables into dummy variables

Scaled numerical data between 0 and 1

Balanced features using standard scaling practices

⚙️ Machine Learning Models Used
Model	Summary
Logistic Regression	Simple linear classifier for baseline comparison
Support Vector Machine (SVM)	Linear SVM for maximizing classification margin
Random Forest Classifier	Ensemble method with decision trees for improved accuracy
Voting Classifier	Ensemble model combining predictions of Logistic, SVM, and Random Forest

📈 Performance Metrics
Accuracy, Precision, Recall, and F1 Score calculated for each model

Confusion Matrices to visualize classification performance

Feature Importance plotted for Random Forest and coefficients visualized for linear models

Side-by-side metric comparison using grouped bar plots

📊 Results Summary

Voting Classifier provided balanced performance with high accuracy and recall.

Visualizations clearly display the comparative strengths and weaknesses of each model.

🖥️ Technologies Used
Language: Python

Libraries: pandas, numpy, matplotlib, scikit-learn

Models: SVM, Logistic Regression, Random Forest, Voting Classifier

💡 Project Highlights
✅ Cleaned and preprocessed real-world structured data
✅ Built, tuned, and compared multiple ML models
✅ Visualized key performance metrics and model insights
✅ Delivered clear business insights on loan approval criteria

📌 Potential Improvements
Implement NLP features from the Text column

Explore hyperparameter tuning with GridSearchCV

Address potential overfitting of Random Forest

Deploy model via a web app using Flask or Streamlit

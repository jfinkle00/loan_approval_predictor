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

<img width="560" height="435" alt="image" src="https://github.com/user-attachments/assets/a16f4eab-9806-44c0-ab23-bd247b1d8ad2" />
<img width="653" height="413" alt="image" src="https://github.com/user-attachments/assets/fba1d0b7-7e70-4541-8624-d74682ecc2c7" />

Income	Credit_Score	Loan_Amount	DTI_Ratio
count	24000.000000	24000.000000	24000.000000	24000.000000
mean	110377.552708	575.720333	44356.154833	34.719167
std	51729.677627	159.227621	34666.604785	32.322471
min	20001.000000	300.000000	1005.000000	2.530000
25%	65635.750000	437.000000	16212.000000	14.507500
50%	110464.000000	575.000000	35207.000000	24.860000
75%	155187.000000	715.000000	65622.750000	41.840000
max	200000.000000	850.000000	158834.000000	246.330000




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

Feature Importance

<img width="556" height="631" alt="image" src="https://github.com/user-attachments/assets/b5926e97-1a87-466f-bb53-485f6d093cdd" />
<img width="559" height="633" alt="image" src="https://github.com/user-attachments/assets/cfb354d3-368f-4529-9fc1-4b58794fc56e" />
<img width="546" height="631" alt="image" src="https://github.com/user-attachments/assets/447ffebb-6cd1-484a-ae08-a309dc7cb077" />




Voting Classifier provided balanced performance with high accuracy and recall.

Visualizations clearly display the comparative strengths and weaknesses of each model.

<img width="846" height="547" alt="image" src="https://github.com/user-attachments/assets/2a1b6ccd-44f1-413a-b3c2-f80054bd84f5" />


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

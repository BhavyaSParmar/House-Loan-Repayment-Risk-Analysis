# House-Loan-Repayment-Risk-Analysis
The project analyzes housing loan repayment risk using data from Kaggle, applying models like Logistic Regression, Decision Trees, and Random Forest. Random Forest outperformed others in predicting loan defaults.

Goal: 
The core goal of this initiative is to craft and deploy sophisticated predictive models capable of accurately predicting challenges in loan repayment. Utilizing the extensive data housed within the “Housing Loan Repayment Risk Analysis” dataset, our objectives include: - Elevating the precision and efficiency of credit risk evaluations in the finance industry. - Uncovering crucial indicators of loan repayment challenges through advanced analytics and machine learning. - Diminishing loan default rates by equipping financial entities with more accurate risk assessment tools. - Refining the loan approval decision-making process to be more knowledgeable and fairer. This project is set to transform the practices of credit risk management fundamentally, bolstering the lending system's stability and trustworthiness.

Methodology
1.	Data Preprocessing and Cleaning: To ensure the “Housing Loan Repayment Risk Analysis” dataset's accuracy and relevance, the initial step will involve cleaning and preprocessing the data. This process will include imputation of missing values, standardization of formats and data types, detection and correction of anomalies or outliers, and ensuring the data is in a state ready for analysis. 

2.	Exploratory Data Analysis: We will engage in exploratory data analysis (EDA) using tools such as correlation matrices, scatterplots, and histograms to uncover patterns and trends within the loan data. This phase will involve visual representation of data through various graphs and charts and the application of statistical techniques to identify correlations between different factors, such as financial background, employment status, and loan specifics (e.g., amount, type, and repayment terms). 

3.	Feature Engineering: To enhance the precision of our predictive models, we will create new variables or feature selection based on domain knowledge and statistical methods to identify significant predictors of loan repayment challenges. Key factors likely to influence our feature selection include applicants' financial backgrounds, employment status, loan amount, annuity, type of loan, repayment terms, and credit history. This step is pivotal in refining our models to accurately predict loan repayment risks. We will use methods like one-hot encoding on the categorical columns like employment status, type of loan which will help in accuracy of the model.

4.	Cross-validation: Is a technique to ensure robust model performance. The dataset is divided into equal parts (folds), and the model is trained and validated across these folds. This method provides a comprehensive assessment of the model's ability to generalize. It further includes:
1.	Training Phase: In this phase, the model learns from the data by adjusting its parameters to predict outcomes based on input features. Typically, about 70-80% of the dataset is used for training, allowing the model to learn the relationship between inputs and outputs.

2.	Testing Phase: In this phase assesses the model's performance on another set of unseen data—the test set. This final evaluation, using about 10-15% of the dataset, provides an unbiased measure of how the model will perform in real-world scenarios.

5.	Model Selection and Evaluation: We plan to utilize machine learning algorithms such as Logistic Regression, Decision Trees, Random Forests to predict loan repayment challenges. These models will be trained on historical data from the “Housing Loan Repayment Risk Analysis” dataset to evaluate their accuracy.

Data Source: https://www.kaggle.com/datasets/urstrulyvikas/house-loan-data-analysis

Dataset size: 158 MB

Results and Analysis:
The result and analysis present a detailed comparison of the performance metrics for three different classification models on a dataset with two classes (0 and 1). Each model's performance is summarized in a classification report that includes precision, recall, f1-score, and support for each class, as well as the overall accuracy and averages for these metrics. 

In-depth analysis of each model's reported metrics:
Logistic Regression:
- Both classes have very similar performance metrics, indicating the model does not significantly favor one class over the other.
- Precision, recall, and f1-score are all above 0.60, which might be considered moderate performance.
- The identical scores across all metrics suggest no particular bias toward precision or recall.
- The overall accuracy is 0.61, meaning that 61% of predictions are correct.

Random Forest:
- Shows a substantial improvement over Logistic Regression.
- High precision for both classes, particularly class '0', indicates that when the model predicts a class, it is highly likely to be correct.
- High recall for class '1' suggests that the model is very good at identifying all relevant instances of that class.
- The f1-scores, which balance precision and recall, are quite high (above 0.90), indicating strong overall performance.
- The overall accuracy is 0.92, so 92% of all predictions are correct.
- Weighted averages are also high, which means the model's performance is consistently good across both classes.

Decision Tree:
- This model has very high precision for class '0' but lower for class '1', suggesting that when it predicts class '0', it is very likely to be correct, but it has more false positives for class '1'.
- The recall is lower for class '0' and very high for class '1', which means it misses some instances of class '0' but captures most of class '1'.
- The f1-score is balanced for both classes, but not as high as for the Random Forest model.
- The accuracy is 0.85, which is lower than Random Forest but still relatively high, meaning 85% of predictions are correct.

Overall Interpretation:
- The Random Forest classifier outperforms the other two in terms of accuracy and balance between precision and recall (f1-score).
- The Logistic Regression model shows moderate performance and might benefit from further parameter tuning or feature engineering.
- The Decision Tree model, while it has a high recall for class '1', might be overfitting to that class and underperforming for class '0' in comparison to the Random Forest model.

In conclusion, Random Forest emerges as a superior model compared to decision trees and logistic regression for predicting house loan defaulters due to its robustness, accuracy, and generalization ability. Unlike decision trees, Random Forest mitigates overfitting by aggregating predictions from multiple trees trained on random subsets of the data, thus capturing diverse patterns and enhancing model stability. Additionally, Random Forest outperforms logistic regression by handling nonlinear relationships and interactions among features more effectively. Its ensemble approach not only provides better predictive performance but also offers resilience against outliers and noisy data. Overall, Random Forest stands out as the optimal choice for lenders seeking accurate and reliable predictions of house loan defaulters, empowering them to make informed decisions and effectively manage lending risks.


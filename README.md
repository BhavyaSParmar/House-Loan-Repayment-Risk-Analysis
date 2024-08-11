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


Description of Dataset
This project focuses on utilizing the “Housing Loan Repayment Risk Analysis” dataset to develop predictive models aimed at improving the precision of credit risk assessments within the finance industry. The dataset includes comprehensive loan application data, such as financial background, employment status, family circumstances, and loan specifics of thousands of applicants. 
The important parameters from the dataset are:

1.	Applicant Financial Background: This column contains detailed information about the loan applicant's financial history, including income level, existing debts, and overall financial health. It's crucial for assessing the applicant's ability to repay the loan.
2.	Employment Status: Indicates whether the applicant is employed, unemployed, self-employed, or retired, providing insights into their stable income sources.
3.	Family Circumstances: Includes data on marital status, number of dependents, and living situation.
4.	Loan Amount: The total amount of money requested by the applicant. This figure is key to determining the scale of the loan and its repayment structure.
5.	Annuity: The regular payment amount the applicant is obligated to pay towards the loan. It helps in understanding the loan's impact on the applicant's monthly budget.
6.	Type of Loan: Specifies the loan's purpose, such as mortgage, personal loan and auto loan.
7.	Repayment Terms: Details of the length of the loan and any specific repayment conditions. It's critical for understanding the loan's duration and the borrower's long-term financial commitment.
8.	Credit History: A record of the applicant's past borrowing and repayments. This column is vital for assessing the borrower's reliability and predicting their future loan repayment behavior.  
These are the real features in the dataset:
NAME_CONTRACT_TYPE
CODE_GENDER
FLAG_OWN_CAR
AMT_INCOME_TOTAL
AMT_CREDIT
AMT_ANNUITY: 
AMT_GOODS_PRICE
NAME_TYPE_SUITE
NAME_INCOME_TYPE
NAME_EDUCATION_TYPE
NAME_FAMILY_STATUS
CNT_CHILDREN
NAME_HOUSING_TYPE
REGION_POPULATION_RELATIVE
DAYS_BIRTH
DAYS_EMPLOYED
DAYS_REGISTRATION
TARGET

Data Source: https://www.kaggle.com/datasets/urstrulyvikas/house-loan-data-analysis

Dataset size: 158 MB

 


When we execute df.info() on a DataFrame df, it prints out information about the DataFrame, including:
Observation:
1.	Range Index: The range of row labels i.e. 307511
2.	Data columns: The column names and data types of each column.
3.	Non-null counts: The count of non-null values for each column.
This method is particularly useful for getting a quick overview of the DataFrame's structure, data types, and memory usage, especially when working with large datasets.

After getting an overview of the dataset we can now deep dive into understanding the data to get our target variable estimation which is our goal.
We will now proceed with the methodologies.

Data Cleaning and Preprocessing:
In order to preprocess and clean the raw data we do the following:

1.	Missing Values: 
The output we got shows that some columns have fewer non-null values than others. For example, the AMT_ANNUITY column has 307499 non-null values, and the AMT_GOODS_PRICE              column has 307233 non-null values out of 307511 total rows. 
These columns have only a few rows of missing values so we decided to drop them.

2.	Exploratory Data Analysis:
This step would involve visually and statistically exploring the dataset to gain insights into its structure, patterns, and relationships between variables. Techniques include summary statistics, data visualization (e.g., histograms, scatter plots, box plots), and correlation analysis.

We found the correlation between different features and plotted the graph between them.
For example: Below we find that the AMT_ANNUITY is highly correlated to AMT_CREDIT

 

Exploratory Data Analysis (EDA):
Exploratory Data Analysis (EDA) is a crucial step to understand the characteristics of the data and identify potential patterns, relationships, and issues that may impact the model's performance. Based on the provided columns, here's a short paragraph on EDA for this dataset:

EDA should focus on examining the distribution and relationships among numerical features like: AMT_CREDIT, DAYS_BIRTH, AMT_ANNUITY.  Visualizations such as histograms, box plots, and scatter plots can reveal skewness, outliers, and correlations. For categorical features like NAME_INCOME_TYPE, NAME_EDUCATION_TYPE, bar plots and countplots can explore their distributions and potential impact on house loan default prediction. 


Categorical Variables:
In categorical variable EDA, we analyze distribution and relationships. Using visualizations like bar plots, we explore frequency distribution, identify outliers, and understand patterns. Cross-tabulation and heatmaps reveal interactions between categories. Bar plots and box plots help assess the relationship with the target variable. We address missing values, outliers, and consider variable transformations. Statistical tests determine significance, guiding subsequent analysis. These insights inform feature engineering and model building.

 


Analysis:  Here we find that the people in the Working category have more applications than any other type.

 




Analysis:  Majority of the loan applicants have higher/ Secondary education with them.

Correlation between different columns

 









Feature Engineering
Now we would be transforming existing features to improve the performance of machine learning models. We will compare many different columns to see which when combined will help us predict the home loan defaulter more accurately. The goal is to extract relevant information from the raw data and create input features that enhance the predictive power of the models.

After cleaning the dataset and understanding the data we will now prepare the dataset in a way so that when we run the dataset under a model, we would be able to get the best results.

1.	Label encoding
Label encoding is a pivotal aspect of feature engineering, particularly for preparing categorical variables for machine learning models. By converting categorical data into numerical format, label encoding enables seamless integration of these features into the model's training process. It assigns unique numerical labels to each category, preserving their ordinality and facilitating interpretation. This transformation enhances model performance by allowing it to effectively learn from categorical information and capture underlying patterns in the data. Overall, label encoding plays a crucial role in enriching the feature space and optimizing the model's predictive capabilities.

2.	Combining columns
We will combine the input numerical columns and the encoded categorical columns to train and find the relationship between the columns.
We find there are 35 categorical columns that we can use to find the relationship and train our model for optimal results.

3.	Balance the dataset
We find that the output target of the dataset is highly imbalanced. The loan defaulter are less than 10% from total applications
 
To solve this problem, we made use of 3 parts of the sample from the target 0 values and target 1 value and balanced the dataset with total 148824 samples.
 

4.	Scaling the numerical columns
We have scaled the numerical columns such as AMT_CREDIT, AMT_INCOME as they have higher value which will affect the model by assigning higher weights to these columns and will lead to incorrect results.

5.	Splitting the dataset into Train and Test set
To test the model accuracy and precession we have divided our dataset into train and using train_test_split module from sklearn library.

Model Selection and Evaluation:
Finally, we would be choosing appropriate machine learning algorithms for the predictive task based on the dataset characteristics and problem requirements. Techniques include evaluating various algorithms and selecting the best-performing ones based on performance metrics and finding which algorithm would give the best scores that would help us predict the home loan defaulters that are listed in the dataset. 
Each of these methodologies plays a crucial role in the predictive modeling process, from data preparation and exploration to feature engineering and model selection. By systematically applying these techniques, we can develop accurate and reliable predictive models.

1.	Logistic Regression
Leveraging the linear relationship between predictor variables and the target variable, linear regression aims to estimate the coefficients of these variables to form a linear equation that best fits the data and converting it into the probably using sigmoid function to classify as defaulter.
In the context of home loan defaulter risk prediction, predictor variables typically include features like annuity, income earned, etc. By fitting a linear regression model to historical loan data, the algorithm learns the relationships between these features, enabling it to make predictions for new instances. Additionally, techniques such as feature scaling, regularization, and feature selection are often applied to enhance model performance and mitigate overfitting. 





Results:
 


 

Results: We achieved max f1 score of 0.62 using Logistic Regression





2.	Decision Tree:
Decision trees are powerful tools used in machine learning for predictive modeling, and they can play a crucial role in predicting house loan defaulters. By analyzing historical data on loan applicants and their repayment behaviors, decision trees can identify patterns and factors that are predictive of whether a borrower is likely to default on their house loan. These factors may include variables such as credit score, income level, employment status, debt-to-income ratio, and past credit history. By constructing a decision tree model based on these factors, lenders can assess the risk of potential borrowers and make more informed decisions on whether to approve a loan application. Additionally, decision trees provide transparency in the decision-making process, as they can illustrate the sequence of factors that lead to the prediction of loan default, helping lenders understand the underlying reasons behind the predictions and enabling them to take appropriate risk management measures. Overall, decision trees serve as valuable tools for lenders in accurately predicting house loan defaulters and mitigating financial risks associated with lending. 
 










3.	Random Forest
Random Forest, a powerful ensemble learning technique, significantly contributes to predicting house loan defaulters by leveraging the collective intelligence of multiple decision trees. By constructing a multitude of decision trees on random subsets of the training data and aggregating their predictions, Random Forest effectively mitigates the risk of overfitting and enhances the model's generalization ability. Each decision tree in the Random Forest learns from different features and instances, capturing diverse patterns and nuances in the data. This diversity among the trees helps in identifying complex relationships and critical factors influencing loan default, such as credit history, income stability, and debt-to-income ratio. Furthermore, Random Forest provides robustness against outliers and noisy data, improving the reliability of predictions. By combining the strengths of individual decision trees, Random Forest delivers accurate and stable predictions of house loan defaulters, empowering lenders to make informed decisions and manage lending risks effectively.


 

 

Hyperparameter tuning:
Hyperparameter tuning is a crucial step in optimizing the performance of decision tree models for predicting house loan defaulters. By adjusting hyperparameters such as the maximum depth of the tree, minimum samples required to split a node, and the criterion for splitting, the model's predictive accuracy can be enhanced. Techniques like grid search or random search can be employed to systematically explore different combinations of hyperparameters and identify the optimal configuration that minimizes prediction errors. For example, increasing the maximum depth of the tree may capture more complex relationships in the data but also risks overfitting, while setting a higher minimum samples per leaf node may prevent overfitting by enforcing a more generalizable model. Through hyperparameter tuning, decision tree models can be fine-tuned to achieve the best possible performance in predicting house loan defaulters, thus aiding lenders in making more accurate risk assessments and informed lending decisions.



 

Decision Tree with max_depth of  2



Results:

 


 

Result: Random Forest achieved the best F1 score of 93%

ROC Curve:
The Receiver Operating Characteristic (ROC) curve is a graphical representation used to evaluate the performance of a classification model, such as Random Forest, across different thresholds. It plots the true positive rate (sensitivity) against the false positive rate (1 - specificity) at various threshold settings.

AUC stands for Area Under the ROC Curve, and it quantifies the overall performance of the model. An AUC value of 0.65 suggests that the Random Forest model has moderate discriminatory power in distinguishing between the positive and negative classes.

In the context of predicting house loan defaulters, an AUC of 0.65 indicates that the Random Forest model is better than random guessing but may still have room for improvement. It means that there is a 65% chance that the model will rank a randomly chosen positive instance higher than a randomly chosen negative instance.

Interpreting the ROC curve with an AUC of 0.65, you would typically see the curve bending towards the top-left corner of the plot, indicating that the model is making meaningful predictions. However, it might not be achieving optimal performance, and further optimization or feature engineering may be necessary to improve the AUC score.
 


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

Support:
- The number of occurrences for each class is provided in the 'support' column, with both classes having a similar number of instances in the dataset. This similarity in support helps in comparing the performance across classes.

Overall Interpretation:
- The Random Forest classifier outperforms the other two in terms of accuracy and balance between precision and recall (f1-score).
- The Logistic Regression model shows moderate performance and might benefit from further parameter tuning or feature engineering.
- The Decision Tree model, while it has a high recall for class '1', might be overfitting to that class and underperforming for class '0' in comparison to the Random Forest model.


Conclusion
In conclusion, Random Forest emerges as a superior model compared to decision trees and logistic regression for predicting house loan defaulters due to its robustness, accuracy, and generalization ability. Unlike decision trees, Random Forest mitigates overfitting by aggregating predictions from multiple trees trained on random subsets of the data, thus capturing diverse patterns and enhancing model stability. Additionally, Random Forest outperforms logistic regression by handling nonlinear relationships and interactions among features more effectively. Its ensemble approach not only provides better predictive performance but also offers resilience against outliers and noisy data. Overall, Random Forest stands out as the optimal choice for lenders seeking accurate and reliable predictions of house loan defaulters, empowering them to make informed decisions and effectively manage lending risks.


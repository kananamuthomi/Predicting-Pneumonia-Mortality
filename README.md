# Predicting-In-Hospital Mortality.
A classification analysis to predict mortality rate of Pneumonia patients.
# Problem Statement.
Healthcare providers and administrators in New York State aim to improve patient outcomes and reduce in-hospital mortality among pneumonia patients. In-hospital mortality occurs when patients succumb to their condition during their hospital stay, and it is crucial for healthcare providers to predict which patients are at the highest risk of mortality. By identifying these high-risk patients in advance, healthcare providers in New York can implement targeted interventions and allocate resources more effectively to improve patient care and survival rates.
# Objectives.
The primary objective of using this dataset is to develop a predictive model that can accurately forecast in-hospital mortality among pneumonia patients. By leveraging the detailed information in the dataset, we aim to:

 - Identify High-Risk Populations: Determine demographic and clinical factors associated with higher pneumonia hospitalization rates.

 - Evaluate Quality of Care: Assess metrics such as length of stay, patient disposition, and mortality rates to gauge the quality of pneumonia care.

 - Assess Economic Burden: Investigate the costs associated with pneumonia hospitalizations and identify opportunities for cost reduction.

 - Reduce Readmissions: Identify factors contributing to pneumonia-related readmissions and suggest strategies to minimize them.
# Data Understanding.
## Data Overview.
The dataset being used for this analysis is the Statewide Planning and Research Cooperative System (SPARCS) Inpatient De-identified dataset. This dataset contains detailed discharge-level information on patient characteristics, diagnoses, treatments, services, charges, and costs for pneumonia patients in New York State from 2009 to 2017. The dataset is de-identified in compliance with the Health Insurance Portability and Accountability Act (HIPAA), ensuring that the health information is not individually identifiable. Direct identifiers have been redacted, such as removing the day and month portions from dates.

### Key Features

- Patient Characteristics: Information about patients such as age, gender, race, and ethnicity.

- Diagnoses: Codes and descriptions of the primary and secondary diagnoses recorded during the hospital stay.
 
- Treatments and Procedures: Details on the treatments and medical procedures performed on the patients.

- Services: Information on the healthcare services provided during the hospital stay.

- Charges and Costs: Financial data related to the total charges and costs incurred during the hospital stay.
### Data Specifics

- Time Period: The dataset spans from 2009 to 2017.

- Geographical Scope: The data covers pneumonia patients in New York State.

- Patient Privacy: All personal identifiers have been removed to protect patient privacy. For example, dates have been modified to exclude the day and month, leaving only the year.
  
## Data Cleaning. 

In the data preparation phase, several steps were undertaken to ensure the dataset's readiness for analysis. These steps included handling missing values using imputation with mean values or a placeholder like 'unknown', removing duplicate records to maintain data integrity, and correcting data entry errors, inconsistencies, or anomalies to ensure accuracy. Additionally, data type inconsistencies were standardized, placeholders were removed or replaced with meaningful values, and numerical features were normalized to a common range. Outliers were identified and appropriately managed to prevent skewing of the analysis, ensuring a robust and accurate dataset for modeling.
## Steps/ Methods.
The project will follow the following steps.

 a) Exploratory Data Analysis -  We will perform an in-depth exploration of the dataset
to gain insights into the distribution of variables, identify patterns, and detect any
data quality issues.

 b) Data Preprocessing: This step involves handling missing values, encoding
categorical variables, and scaling numerical features.

 c)  Feature Selection: We will identify relevant features that have a significant impact
on customer churn prediction.

 d)  Model Selection and Training: We have used both machine learning (Logistic Regression and Decision Tree) as well as Deep Learning models (Feed Forward Neural Network and Multilayer Perceptron, Ensemble methods and Artificial Neural network to predict in hospital mortality rate.
 
 e) Model Evaluation: We will assess the performance of the trained model using
appropriate evaluation metrics, including accuracy, precision, recall, and F1-score.

 f) Model Optimization: We will fine-tune the selected model by adjusting
hyperparameters and employing techniques like grid search. This optimization
process aims to maximize the model's predictive capabilities. 

 ## Exploratory Data Analaysis.
![image](https://github.com/user-attachments/assets/9f180ae0-9c12-4c90-88d0-76c2d46f2188)

The histograms of the numeric columns reveal varying distributions. Some columns exhibit skewed distributions, indicating that data is not evenly spread and may be concentrated towards one end of the range. Other columns show more symmetric distributions, suggesting a more uniform spread of values.

![image](https://github.com/user-attachments/assets/f0d787da-3d3f-4788-914f-7dcff250564a)
### Interpretation

No of patients are quietly more in New York city than other cities.

More number of patients fall in the age category 70 years or older.

Female patients are more in number as comapre to male patients.

Patients enrolled in Emergency are huge in number.

Patient Disposition i.e. patients destination after discharge, mostly are in Home or Self Care prescription.

Mostly patients have Minor Risk of Mortality.

Patients have Medicare payment typology more

![image](https://github.com/user-attachments/assets/36eac9cc-e498-43e8-adb9-63198f24ab4b)
The graph shows the top 10 facilities based on the number of records, with Montefiore Medical Center having the highest number of records.

![image](https://github.com/user-attachments/assets/5ccfd41b-73b6-4037-8884-19a119975b81)
The most prevelant Zip Code is 122

![image](https://github.com/user-attachments/assets/1d42840a-673d-4c54-9ecc-f01a96ec4444)
The top 10 lengths of stay predominantly range from 1 to 10 days, with shorter stays being much more common. In contrast, the bottom 10 lengths of stay, which are significantly longer, are rare, indicating that extended stays are exceptional in the dataset.

![image](https://github.com/user-attachments/assets/cbadc987-c818-41ee-b00d-1d2568bfd48b)
The most dominant CCS diagnosis is Pneumonia that is either caused by tuberculosis or sexually transmitted diseases.

![image](https://github.com/user-attachments/assets/a70d49fb-6778-4401-8f4b-ccdd06211aa7)
In the dataset, "NO PROC" is the most common CCS Procedure Description, indicating that no procedure was performed in these cases. In contrast, the least common descriptions, each with only one occurrence, include rare or less frequently performed procedures.

![image](https://github.com/user-attachments/assets/a5f9cdf3-362c-44e2-8868-cdd95b478ba4)
The top 10 APR DRG descriptions primarily focus on pneumonia and respiratory conditions, with "Other pneumonia" and "OTHER PNEUMONIA" being the most common. Conversely, the bottom 10 descriptions are rare and involve specific or complex procedures such as bone marrow and heart transplants, with counts as low as 1 or 2.

![image](https://github.com/user-attachments/assets/4b76d4ef-2390-4119-86c4-9bae53e1aa6b)
The most dominant APR MDC Description is Diseases and Disorders of the Respiratory System.

## Bi Variate Analysis.
### Variables vs Target Variable.
![image](https://github.com/user-attachments/assets/23d31534-e56a-4c3f-a96c-8713fde80a26)

![image](https://github.com/user-attachments/assets/d2f0441a-bdbc-498e-bcca-6c581e995d71)

### Gender Vs Other Features
![image](https://github.com/user-attachments/assets/f285afca-54f6-4e9c-a5ea-74200f0b05b6)

![image](https://github.com/user-attachments/assets/3e181fbb-3634-4c57-8043-6c219e245fd8)

## Total Costs Vs Other features
![image](https://github.com/user-attachments/assets/8ddea9fe-c32c-4b68-be7b-3e3de4548b58)

## Total Charges Vs Other features.
![image](https://github.com/user-attachments/assets/1c8876f4-fb68-48e5-a37f-9b9cb432279d)

## 3.3 Multivariate Analysis
### Severity of Ilness by Gender and Race.
![image](https://github.com/user-attachments/assets/0e780703-5337-4a3a-8f17-16f1455f3431)

 - Severity Distribution: Females generally have higher counts in the major and moderate severity categories, while males have slightly higher counts in the extreme severity category.
 - Gender Distribution: For all severity levels, the number of cases with unknown gender is very low, indicating that the majority of cases have well-documented gender information.
 - Comparative Insight: Most severity levels show more cases for females compared to males, except for extreme severity, where males slightly outnumber females.
   
![image](https://github.com/user-attachments/assets/848c0ba6-cf38-4b23-abf0-575e8e2356ab)

## Colinearity and Multi Colinearity.
Checking for collinearity between dependent and independent variables.
![image](https://github.com/user-attachments/assets/135bd69a-3536-4681-90fd-29d623db9311)
Checking for Multicolinearity High correlation,(e.g above 0.7)indicates multicollinearity. From the dia gram above, the total charge and total charges had multicolinearity.
We will therefore use PCA to deal with multicolinearity as they are important columns for our analysis and therefore we cannot drop them.
![image](https://github.com/user-attachments/assets/47b17327-4cbe-4fce-8599-4407be337b6c)
PCA reduces the variables into principal components that reduces multicolinearity

# 4. Modelling
## 4.1 Modelling Processing
We explored both traditional machine learning models and deep learning models to predict the target variable, which is a multi-class classification problem with four distinct classes.

Machine Learning Models
- Logistic Regression Model(Baseline)
- Decision Tree
- XG Boost

Deep Learning Models
For deep learning models, we employed several architectures to leverage their capacity to learn complex patterns from data:

- Feedforward Neural Network (FNN)
- Multilayer Perceptron (MLP)
- Artificial Neural Network (ANN)
- Ensemble Methods (DNN, ANN, and MLP)

**Optimizer and Loss Function**

Optimizer: Adam

The Adam optimizer was chosen for training our deep learning models. Adam is an adaptive learning rate optimization algorithm that combines the advantages of two other popular methods: AdaGrad and RMSProp. It is well-suited for handling sparse gradients on noisy data. Its key features include:
- Adaptive Learning Rates: Adam adjusts the learning rate for each parameter individually, which is beneficial for models with complex parameter spaces.
- Efficient: Requires less memory and is computationally efficient.
- Robust: Works well in practice for a wide range of deep learning architectures and problems.

Loss Function: Sparse Categorical Crossentropy

We used sparse_categorical_crossentropy as the loss function for our multi-class classification task. The reasons for choosing this loss function include:

- Efficient for Multi-Class Problems: Specifically designed for multi-class classification where the target variable can belong to one of several classes.
- Minimizes the Log Loss: Encourages the model to produce probabilities close to 1 for the true class and close to 0 for other classes, optimizing classification accuracy.
## 4. 2 Evaluation Metrics

For evaluating the performance of our models, we will focus primarily on recall and precision, while also considering f1 score, confusion matrix and ROC Curve as supplementary metrics.

### Primary Metrics

**Recall:**

Recall is crucial in predicting "APR Risk of Mortality" because it measures the model's ability to correctly identify high-risk patients. Given the serious consequences of missing a high-risk case (false negative) in a healthcare setting, our goal is to achieve a recall of 85%. This target will serve as the benchmark for determining whether the model performs well in capturing true positive cases.

**Precision:**

Precision indicates the model’s ability to provide accurate predictions and minimize false positives. In healthcare, high precision ensures that when the model identifies a patient as high-risk, there is a strong likelihood that the patient is indeed high-risk. While our primary focus is on achieving a recall of 85%, maintaining high precision is also important to avoid unnecessary interventions or treatments.

**Balance between Recall and Precision**

Achieving a balance between recall and precision is essential. Our goal is to maximize recall while maintaining acceptable precision, thereby minimizing both false negatives (missed high-risk cases) and false positives (incorrectly classified low-risk cases).

### Additional Metrics

**Confusion Matrix:**

The confusion matrix will provide detailed insights into the types of errors the model makes. It will help us understand where the model struggles and which classes are often confused with each other, guiding further improvements.
Our target recall of 85% will serve as the baseline for assessing the model’s performance, with other metrics providing additional context and insights.

**ROC Curve**

It's an essential tool for evaluating the performance of classification models. It provides a comprehensive view of a model's ability to distinguish between different classes and helps in making informed decisions about threshold selection and model improvements.The model with the highest AUC is the best model to use for feature importance.

**Feature Importance**

With the highest performing model, we were able to conduct feature importance and identify the most important features that predict In hospital mortality for pneumonia patients.

## 4.3 Data Preprocessing
### Feature Selection and Target Variable

Features: We used the principal components to train our machine learning models as it takes a shorter time. All columns from the dataset were used as features to build our deep learning models. This is because the deep learning models could handle the large dataset and gave us the best results. 

Target Variable: The target variable for our classification task is the "APR Risk of Mortality," which is a multi-class variable with four distinct classes representing different levels of mortality risk.

### Data Usage Strategy

Due to the large size of our dataset, we adopted different data usage strategies for different models:

**Logistic Regression, Decision Trees and XG Boost:**

For this models, we used PCA to obtain principal componenets then trained our data on this using this models. This approach was chosen to manage computational resources efficiently, as decision trees can be memory-intensive and computationally expensive when trained on very large datasets. 

**Other Models (XG Boost, FNN, MLP, ANN, Ensemble Methods):**

For these models, we utilized the entire dataset. The full dataset allows these models to fully capture the patterns and complexities present in the data. Deep learning models, in particular, benefit from larger datasets due to their capacity to learn intricate patterns and dependencies. XG Boost was identified as the best model from the ROC Curve, we then trained it using the entire dataset and then used it for feature importance.

**Logistic Regression**
1) For the Extreme class, the model has a relatively higher precision but very low recall. This indicates that when the model predicts the Extreme class, it is correct 57% of the time, but it only identifies 12% of the true Extreme cases.

2) The Major class has moderate precision and recall, meaning that the model predicts this class with some degree of accuracy but still misses a considerable number of actual cases.

3)The Minor class has the highest precision, recall, and F1-score among all classes, indicating that the model performs best in predicting this class.

4) For the Moderate class, the precision and recall are slightly below average, suggesting that the model has difficulty distinguishing this class from other.

**Decision Tree**
The overall accuracy of the decision tree model is 43.78%, meaning the model correctly classified approximately 44% of the samples.

1) Out of all predictions made for the "Extreme" class, only 26% were correct. This indicates a relatively high number of false positives. Of all actual "Extreme" instances, the model correctly identified 27%. This reflects a considerable number of false negatives.

2)The model's precision for the "Major" class is 38%, showing a high number of false positives.The recall is 38%, indicating a significant number of false negatives.

3)For the "Minor" class, 55% of the predictions were accurate. This is the highest precision among the classes. The model correctly identified 56% of the actual "Minor" cases, which is comparatively better than other classes.

4)The precision for the "Moderate" class is 40%, showing room for improvement. The recall is 40%, indicating a need to reduce false negatives.

**XG Boost**
1) Out of all predictions made for the "Extreme" class, 50% were correct. This indicates that half of the positive predictions are true positives. For Recall, the model correctly identified only 20% of the actual "Extreme" instances, which means there are a significant number of false negatives.

2)The model's precision for the "Major" class is 49%, showing a moderate number of false positives.The recall is 51%, indicating the model captures slightly more than half of the actual "Major" cases.

3)For the "Minor" class, 63% of the predictions were accurate, making this the class with the highest precision. The model correctly identified 71% of the actual "Minor" cases, demonstrating strong recall performance.

4)The precision for the "Moderate" class is 47%, showing that there are some false positives. The recall is 46%, suggesting a number of false negatives.

**DEEP LEARNING MODELS**
### Feed Forward Neural Network
### Interpretation

**Class 0 (Extreme):**

Precision (0.68): Of all instances predicted as Class 0, 68% are actually Class 0.

Recall (0.57): Of all actual Class 0 instances, 57% are correctly identified as Class 0.

F1-Score (0.62): Balances precision and recall for Class 0.

Class 0 has moderate precision but lower recall. This indicates that while the predictions for Class 0 are reasonably accurate, the model misses a significant number of true Class 0 instances. The F1-Score reflects this balance, showing that improvements could be made in correctly identifying more Class 0 instances.

**Class 1 (Major):**

Precision (0.60): Of all instances predicted as Class 1, 60% are actually Class 1.

Recall (0.57): Of all actual Class 1 instances, 57% are correctly identified as Class 1.

F1-Score (0.58): Balances precision and recall for Class 1.

Class 1 has similar precision and recall, indicating moderate performance. However, the model struggles to accurately identify 
Class 1 instances, as reflected by the lower F1-Score. The number of false negatives (misclassified instances) for Class 1 is relatively high.

**Class 2 (Moderate):**

Precision (0.64): Of all instances predicted as Class 2, 64% are actually Class 2.

Recall (0.68): Of all actual Class 2 instances, 68% are correctly identified as Class 2.

F1-Score (0.66): Balances precision and recall for Class 2.

Class 2 has a good balance between precision and recall, with the highest recall among the classes. This suggests that the model performs relatively well in identifying Class 2 instances. The F1-Score is also reasonably high, indicating effective overall performance for Class 2.

**Class 3 (Minor):**

Precision (0.84): Of all instances predicted as Class 3, 84% are actually Class 3.

Recall (0.83): Of all actual Class 3 instances, 83% are correctly identified as Class 3.

F1-Score (0.84): Balances precision and recall for Class 3.

Class 3 exhibits the highest precision and recall, making it the best-performing class in terms of both correctly identifying and predicting true positives. The high F1-Score confirms strong performance for this class, with few misclassifications.

**Overall Metrics**
Accuracy (0.70): The model correctly classifies 70% of all instances across all classes. This is a general measure of model performance.

**Overall Insights**

Class 3 performs the best with high precision and recall, indicating the model is very effective in predicting this class.

Class 2 also performs well but has slightly lower precision compared to Class 3.

Class 0 and Class 1 show lower performance, particularly in recall, indicating the model struggles to correctly identify these classes.

### 2. Multi Layer Perceptron.
### Interpretation

**Class 0 (Extreme):**

Precision (0.68): Of all instances predicted as Class 0, 68% were correctly classified.

Recall (0.61): Of all actual Class 0 instances, 61% were correctly identified.

F1-Score (0.64): The harmonic mean of precision and recall, balancing both metrics for Class 0.

Class 0 has a moderate precision and recall. The F1-Score is decent but indicates that there is room for improvement in both precision and recall. The support shows a relatively smaller number of instances compared to other classes.

**Class 1 (Major):**

Precision (0.60): Of all instances predicted as Class 1, 60% were correctly classified.

Recall (0.61):Of all actual Class 1 instances, 61% were correctly identified.

F1-Score (0.60): Balances precision and recall for Class 1.

Class 1 has similar precision and recall values, indicating a balanced performance but relatively lower compared to Class 2 and Class 3. The support is high, meaning Class 1 instances are more frequent in the dataset, but the model's performance on this class is less strong.

**Class 2 (Moderate):**

Precision (0.65):Of all instances predicted as Class 2, 65% were correctly classified.

Recall (0.66): Of all actual Class 2 instances, 66% were correctly identified.

F1-Score (0.66): Balances precision and recall for Class 2.


Class 2 has a good precision and recall, with a higher F1-Score compared to Class 0 and Class 1. This indicates the model performs fairly well for Class 2. The support is the highest among all classes, suggesting it is a common class in the dataset.

**Class 3 (Minor):**

Precision (0.84): Of all instances predicted as Class 3, 84% were correctly classified.

Recall (0.83): Of all actual Class 3 instances, 83% were correctly identified.

F1-Score (0.84): Balances precision and recall for Class 3.

Class 3 shows the highest performance in terms of precision, recall, and F1-Score. This indicates that the model identifies Class 3 instances very well, with a high degree of accuracy. The support is also high, showing it is a significant class in the dataset.

**Overall Metrics**

Overall Accuracy (0.70): The model correctly predicts 70% of all instances across all classes.

**Overall Insights**

Class 0 and Class 1 have lower precision and recall compared to Class 2 and Class 3, indicating they are less well predicted.

Class 2 and Class 3 are identified more accurately, with Class 3 being the best-performing class in terms of precision and recall.

###  Artifical Neural Network
### Interpretation

**Class 0 (Extreme):**

Precision (0.68): When the model predicts Class 0, it is correct 68% of the time. This indicates a relatively good ability to identify Class 0 correctly.

Recall (0.69): Out of all actual Class 0 instances, the model successfully identifies 69% of them. This suggests that the model performs reasonably well in detecting Class 0 cases.

F1-Score (0.68): The harmonic mean of precision and recall, providing a balanced measure of the model’s performance for Class 0.

**Class 1 (Major):**

Precision (0.62): When the model predicts Class 1, it is correct 62% of the time. This indicates that the model has some difficulty accurately predicting Class 1.

Recall (0.64): Out of all actual Class 1 instances, the model identifies 64% of them. This suggests that the model misses a notable number of Class 1 instances.

F1-Score (0.63): The F1-score shows a balanced performance for Class 1, but it is lower than that for Class 0 and Class 3.

**Class 2 (Moderate):**

Precision (0.67): When predicting Class 2, the model is correct 67% of the time. This shows a fairly good precision for Class 2.

Recall (0.66): The model identifies 66% of the actual Class 2 instances. The recall is slightly lower compared to precision, indicating some missed Class 2 instances.

F1-Score (0.67): The F1-score reflects a balanced performance for Class 2, comparable to Class 0 but slightly better than Class 1.

**Class 3 (Minor):**

Precision (0.85): The model is highly accurate when predicting Class 3, with 85% precision. This indicates excellent performance in predicting Class 3.

Recall (0.84): The model identifies 84% of the actual Class 3 instances, reflecting a high recall rate.

F1-Score (0.84): The F1-score for Class 3 is the highest, demonstrating the model's strong overall performance for this class.

**Overall Metrics**

Accuracy (0.72): Overall, the model correctly classifies 72% of instances across all classes. This is a good accuracy rate, indicating that the model performs well in general.

**Overall Insights**

Class 3 shows the strongest performance with the highest precision, recall, and F1-score, making it the best-predicted class.

Class 0 and Class 2 have moderate performance, with balanced precision and recall.

Class 1 has lower precision and recall compared to other classes, indicating that the model struggles more with predicting this class accurately.

## Ensemble Methods
### Interpretation

**Class 0 (Extreme):**

Precision (0.70): 70% of the time, when the model predicts Class 0, it is correct. 

Recall (0.66): The model correctly identifies 66% of all actual Class 0 instances. There's some room for improvement in capturing more true positives.

F1-Score (0.68): Balances precision and recall, suggesting that the model performs moderately well in predicting Class 0.

**Class 1 (Major):**

Precision (0.62): 62% of predictions for Class 1 are correct. This indicates the model makes a fair amount of false positive errors.

Recall (0.64): The model captures 64% of actual Class 1 instances, showing moderate sensitivity.

F1-Score (0.63): Indicates a need for improvement in both capturing true positives and avoiding false positives for Class 1.

**Class 2 (Moderate):**

Precision (0.67): 67% of predicted instances for Class 2 are correct.

Recall (0.67): The model successfully identifies 67% of actual Class 2 instances.

F1-Score (0.67): Shows balanced performance for Class 2, with similar levels of precision and recall.

**Class 3 (Minor):**

Precision (0.85): High precision indicates that when the model predicts Class 3, it is correct 85% of the time.

Recall (0.84): The model captures 84% of the actual Class 3 instances, showing excellent sensitivity.

F1-Score (0.85): The high score reflects a strong ability to predict Class 3 correctly and efficiently.

**Overall Performance**

Accuracy (0.72): Overall, the model is 72% accurate in its predictions across all classes. 

The model performs well for Class 3 and reaches our target but there are opportunities to enhance recall and precision for the other classes to improve overall performance.

**ROC CURVE**

![image](https://github.com/user-attachments/assets/53929371-24cc-4f7f-8432-3378323aaa10)

XG Boost seems to have the highest AUC scores across all classess. it has the best ability to distinguish between positive and negative classes among the models listed. It performs exceptionally well and can be considered the strongest model.

**Feature Importance**
Since we had initially used principal components, lets train the model on the entire dataset so as to get the features that are most significant to our prediction.
The overall accuracy of the XGBoost model is approximately 72.04%, meaning it correctly classified about 72% of the samples in the dataset.

1)Of all the predictions made for the "Extreme" class, 70% were correct, indicating a relatively balanced rate of false positives.The model correctly identified 67% of actual "Extreme" instances, suggesting a few false negatives.

2) The model's precision for the "Major" class is 62%, showing a moderate number of false positives. The recall is 66%, indicating the model captures most of the actual "Major" cases.

3)For the "Minor" class, 63% of the predictions were accurate, making this the class with the highest precision.The model correctly identified 71% of the actual "Minor" cases, demonstrating strong recall performance.

4) The precision for the "Moderate" class is 47%, showing that there are some false positives. The recall is 46%, suggesting a number of false negatives.

![image](https://github.com/user-attachments/assets/c6081ad4-8df5-45d2-8413-136c834e38c8)
# 5. Conclusion
From the above analysis, we can conclude that;
1) The APR Severity of Illness Code is the most significant predictor of the APR Risk of Mortality meaning. This indicates that the more severe a patient's condition, the higher their risk of mortality.

2) Age is a crucial factor in determining mortality risk. Patients aged 70 or older have a significantly higher risk of mortality. This emphasizes the importance of prioritizing elderly patients for closer monitoring and care.
  
3) Patients undergoing hemodialysis have an increased risk of mortality, highlighting the need for careful management of patients with renal failure or related conditions.

4) The patient's disposition status, especially whether they are discharged to home or self-care or have expired, indicates significant mortality risk factors, reflecting care transitions' outcomes.

5) The descriptions of illness severity, particularly Major and Moderate, also play a significant role in predicting mortality risk. Patients classified as having major or moderate severity levels should be monitored closely.

# 6. Recommendations

1) Enhance Monitoring: Implement advanced monitoring and treatment strategies for patients with higher APR Severity of Illness Codes (Extreme, Major, Moderate). This includes deploying resources for intensive care units and specialized treatment plans.

2) Age-Specific Care Plans: Create age-specific care plans, focusing on preventive measures and early interventions for patients aged 70 and older. This can be done by conducting regular health assessments for elderly patients to identify potential risks early and address them promptly.

3) Comprehensive Care for Renal Patients: Develop comprehensive care programs for patients undergoing hemodialysis, including regular check-ups, nutritional support, and access to specialized care.
   
4)Improve Transion Care - Enhance discharge planning and follow-up care for patients discharged to home or self-care. Ensure that they have access to necessary resources and support to prevent readmissions or adverse outcomes.

5) Healthcare Policy: Advocate for healthcare policies that emphasize preventive care, early intervention, and resource allocation based on severity and age-related risk factors.
   
6) Track Readmissions - The current dataset does not include information on hospital readmissions, making it difficult to identify which demographic groups are more frequently affected. Assigning a unique patient ID to each individual would enable more detailed tracking and provide deeper insights into paterns of readmissions across demographics.


































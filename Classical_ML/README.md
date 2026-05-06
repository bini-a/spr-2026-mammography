# Classical ML Approaches

This project aim to compare classical machine learning approaches for predicting BI-RADS categories from mammography reports

## Data Preprocessing
All reports undergo a baseline cleaning step (e.g lowercasing, removal of anonymization tokens, carriage returns, newlines removal, etc). Beyond this baseline clean data, we evaluate four additional preprocessing techniques:
- Stop word removal using language specific NLTK corpora.
- Stemming using NLTK SnowballStemmer for the target language
- Lemmatization using spaCy lookup lemmatizer

Extracted features using Term Frequency-Inverse Document Frequency (TF-IDF) vectorization at character and word level and  combined them through horizontal concatenation.

Then conduct preprocessing ablation study evaluating six preprocessing configurations using Linear SVC with 5-fold stratified cross-validation on the original Portuguese dataset.

![ablation](/Classical_ML/outputs_pt/plot3_preprocessing_ablation.png)

## Model Comparison
Seven classification algorithms organized into three families were evaluated:
- Linear Models
    - Linear Support Vector Classifier (LinearSVC)
    - Logistic Regression
    - SGD Classifier (modified Huber loss)
- Probabilistic Models
    - Multinomial Naive Bayes
    - Complement Naive Bayes
- Non Linear Models (Gradient Boosting)
    - LightGBM
    - XGBoost
 
Standard 5-fold stratified cross-validation was used whereby in each iteration, 4 folds are used for training and 1 fold for
validation.

![model](/Classical_ML/outputs_pt/plot4_model_comparison.png)

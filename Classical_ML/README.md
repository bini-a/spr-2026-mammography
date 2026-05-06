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

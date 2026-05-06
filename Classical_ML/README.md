# Classical ML Approaches

This project aim to compare classical machine learning approaches for predicting BI-RADS categories from mammography reports

## Data Preprocessing
All reports undergo a baseline cleaning step (e.g lowercasing, removal of anonymization tokens, carriage returns, newlines removal, etc). Beyond this baseline clean data, we evaluate four additional preprocessing techniques:
- Stop word removal using language specific NLTK corpora.
- Stemming using NLTK SnowballStemmer for the target language
- Lemmatization using spaCy lookup lemmatizer

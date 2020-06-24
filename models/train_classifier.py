import sys
from sqlalchemy import create_engine
import numpy as np
import pandas as pd
import pickle

import re
import nltk
nltk.download(['punkt','wordnet'])
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score, fbeta_score, make_scorer


def load_data(database_filepath):
    '''
    Load the data from database
    Input:
        database_filename (str): database filename.
    Output:
        X - Obeservation dataset
        Y - targeting dataset
        category_names - list containing category names
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('df', engine)
    X = df['message']
    y = df.iloc[:,4:]
    category_names = list(df.columns[4:])
    return X, y, category_names


def tokenize(text):
    '''
    Clean and tokenize text
    Input:
        text: original message text
    Output:
        clean_tokens: cleaned, tokenized, and lemmatized tokens
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # Replace possible urls with note
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    # Normalize Text
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
    # Tokenized text
    tokens = word_tokenize(text)
    # Instantiate Lemmatizer
    lemmatizer = WordNetLemmatizer()
    # Lemmatize tokens
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)

    return clean_tokens



def build_model():
    '''
    Build a ML pipeline using ifidf, random forest, and gridsearch
    Input: None
    Output:
        pipeline - pipeline after GridSearchCV
    '''
    pipeline = Pipeline([
                        ('vect', CountVectorizer(tokenizer=tokenize)),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultiOutputClassifier(RandomForestClassifier()))
                        ])

    parameters = {   
            'vect__ngram_range': ((1, 1), (1, 2)),
            # 'vect__max_df': (0.5, 1.0),
            # 'tfidf__use_idf': (True, False),
            # 'clf__estimator__min_samples_split': [2, 3]
        }

    pipeline = GridSearchCV(pipeline, param_grid=parameters, n_jobs=4, verbose=2)
    
    return pipeline



def evaluate_model(model, X_test, y_test, category_names):
    '''
    Evaluate the model performances and print the results
    Input:
        model - pipeline model to evaluate
        X_test - testing observation dataframe
        Y_test - testing targeting dataframe
        category_names - list of category names
    '''
    y_pred = model.predict(X_test)

    for i in range(len(category_names)):
        print("Category:", category_names[i],"\n", classification_report(y_test.iloc[:, i].values, y_pred[:, i]))
        print('Accuracy of %25s: %.2f' %(category_names[i], accuracy_score(y_test.iloc[:, i].values, y_pred[:,i])))


def save_model(model, model_filepath):
    """    
    This function saves trained model as Pickle file.
    
    Input:
        model -> GridSearchCV or Scikit Pipelin object
        model_filepath -> destination path to save .pkl file
    Output:
        None
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    '''
    Main function
    
    1. Load data from database_filepath.db
    2. Train ML model on training set
    3. Estimate model performance on test set
    4. Save trained model as model_filepath.pkl
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
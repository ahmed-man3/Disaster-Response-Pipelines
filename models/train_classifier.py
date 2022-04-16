import sys
from sqlalchemy import create_engine
import pandas as pd
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def load_data(database_filepath):
    # load data from database
    engine = create_engine(database_filepath)
    df = pd.read_sql_table(table_name='messages',con=engine)
    
    # defaine X, y and category names
    X = df['message']
    y = df.iloc[:,4:]
    category_names = y.columns

    return X, y, category_names


def tokenize(text):
    return word_tokenize(text)


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    y_preds = model.predict(X_test)
    i = 0
    for c in category_names:
        
        print(c)
        print(classification_report(Y_test.iloc[:,i],y_preds[:,i]))
        i = i + 1
        
    


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as files:
        pickle.dump(model,files)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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
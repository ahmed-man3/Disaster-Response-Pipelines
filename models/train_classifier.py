import sys
from sqlalchemy import create_engine
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer



def load_data(database_filepath):
    """
    A function to load the data

    We used Pandas function combined with sqlalchemy to read Database file,
    then extracted X, y and column names for categories. 

    Parameters:
    messages_filepath : Sqlite filepath for database.

    Returns:
    X : messages from the dataframe
    y : 36 categories columns values
    category_names : 36 categories columns names
    """
    # load data from database
    engine = create_engine(database_filepath)
    df = pd.read_sql_table(table_name='messages',con=engine)
    
    # defaine X, y and category names
    X = df['message']
    y = df.iloc[:,4:]
    category_names = y.columns

    return X, y, category_names


def tokenize(text):
    """
    A tokenization function to process text data.

    The function uses a custom tokenize function using nltk 
    to case normalize, lemmatize, and tokenize text. 

    Parameters:
    text : a text from DF messages.

    Returns:
    A clean tokens of the text. 
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    A function to build the model

    The script builds a pipeline that processes text and then performs
    multi-output classification on the 36 categories in the dataset.
    GridSearchCV is used to find the best parameters for the model.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'clf__estimator__min_samples_split': [2, 4]
    }

    cv = GridSearchCV(pipeline,param_grid=parameters) 
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    A function to evaluate the model

    Report the f1 score, precision and recall for each output
    category of the dataset. This has been done by iterating through
    the columns and calling sklearn's `classification_report` on each.

    Parameters:
    Model : pipline trined model.
    X_test: messages test list.
    Y_test: 36 categories test list.
    category_names: list of the categories names.
    """
    y_preds = model.predict(X_test)
    i = 0
    for c in category_names:
        
        print(c)
        print(classification_report(Y_test.iloc[:,i],y_preds[:,i]))
        i = i + 1
        
    


def save_model(model, model_filepath):
    """
    A function to save the model

    We used this function to save the model into pickle file.

    Parameters:
    model : pipline trined model.
    model_filepath: filepath for Pickle file to be saved.
    """
    with open(model_filepath, 'wb') as files:
        pickle.dump(model,files)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)

        #Split data into train and test sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        #Train pipeline
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
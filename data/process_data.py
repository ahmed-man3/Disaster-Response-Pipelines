import sys
import pandas as pd
from sqlalchemy import create_engine
import numpy as np

def load_data(messages_filepath, categories_filepath):
    """
    A function to load the data

    We used Pandas function to read CSV files and then merge
    the files. 

    Parameters:
    messages_filepath : CSV filepath for messages list.
    categories_filepath: CSV filepath for categories list.

    Returns:
    merged Dataframe of messages and categories 
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = pd.merge(messages,categories,on='id')

    return df


def clean_data(df):
    """
    A function to clean the data

    To ensure that our Dataframe is cleaned and ready.

    Parameters:
    Dataframe : dataframe of the merged messages and categories 

    Returns:
    Cleaned Dataframe of messages and categories 
    """
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(";", expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[:1]

    # use this row to extract a list of new column names for categories.
    # We applied a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x.str[0:-2])
    category_colnames=category_colnames.iloc[0,:].tolist()

    # rename the columns of `categories`
    categories.columns = category_colnames

    #Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] =  categories[column].astype("str").str[-1:]
        
        # convert column from string to numeric and to binary (0,1)
        categories[column] = np.where(categories[column].astype("int64")<1,0,1)

    #Replace `categories` column in `df` with new category columns.
    df = df.drop(columns='categories')

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)

    # drop duplicates
    df=df.drop_duplicates()

    return df

def save_data(df, database_filename):
    """
    A function to save the cleaned dataframe.

    We did this with pandas `to_sql` method combined with the SQLAlchemy library.

    Parameters:
    Dataframe : dataframe of the cleaned dataframe.
    database_filename: Database file destintion.
    """
    engine = create_engine(database_filename)
    df.to_sql(name='messages', con=engine, index=False, if_exists='replace') 
    


def main():
    if len(sys.argv) == 4:

        
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
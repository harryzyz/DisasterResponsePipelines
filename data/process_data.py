import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
        This function reads messages.csv from messages_filepath and 
        categories.csv from categories_filepath. Then merge them in to a datafrom df

        Input:  messages_filepath - message.csv filepath
                categories_filepath - categories.csv filepath
        
        Output: df
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, left_on='id', right_on='id')
    return df


def clean_data(df):
    '''
    This function corrects the names of category columns 
    and assign integer values accordingly.

    Input:  df
    Output: df
    '''
    categories = df['categories'].str.split(';',expand=True)
    categories.columns = categories.iloc[0].apply(lambda x: x.split('-')[0])
    
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split('-')[-1]).apply(int)

    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories],axis=1)
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    '''
    This function loads df to database_filename.db under current folder.

    Input:  df - dataframe
            database_filename name of .db file

    Output: None
    '''
    engine = create_engine('sqlite:///DisasterResponse.db')
    df.to_sql('df', engine, index=False)


def main():
    """
    Main Data Processing function
    
    This function implement the ETL pipeline:
        1) Data extraction from .csv
        2) Data cleaning and processing
        3) Data loading to SQLite database
    """
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
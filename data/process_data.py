import sys
import pandas as pd 
import numpy as np 
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Function:
        1.Load the data from two csv files (disaster_messages.csv and disaster_categories.csv)
        2.Merging these two data sets into one dataset 
    
    Args:
        messages_filepath (str): filepath of disaster_messages.csv
        categories_filepath (str): filepath of disaster_categories.csv
        
    Returns:        
        df (DataFrame): A dataframe combining messages and categories
    '''
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath) 
    # merge datasets
    
    df = pd.merge(messages, categories, on=['id'], how='left')
    return df
   

def clean_data(df):
    '''
    Function:
        1.Convert the categories of disaster messages into 36 individual category columns
        2.Rename individual category column with the corresponding category name in text
        3.Remove duplicate row
    
    Args:
        df (DataFrame): A dataframe combining messages and categories
        
    Returns:        
        df (DataFrame): A dataframe with messages and with 36 category columns
    '''
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', expand = True)
    
    # select the first row of the categories dataframe
    row = categories.head(1)
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x.str[:-2], axis=1).values.tolist()[0]
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in category_colnames:
        # set each value to be the last character of the string
        categories[column] =  categories[column].astype(str).str[-1]
    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column]).astype(int)
    
    # drop the original categories column from `df`
    df = df.drop(["categories"], axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df=df.drop_duplicates()
    
    return df

def save_data(df, database_filename):
    '''
    Function:
        1.Save the data frame of disaster messages with 36 categories into the DisasterResponse.db
        
    Args:
        df (DataFrame): A dataframe of disaster messages with 36 categories
        database_filename: the name of the database which we save message's dataframes       
        
    '''
     
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')

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
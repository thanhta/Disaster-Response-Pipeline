import sys# import libraries
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
import pickle 

def load_data(database_filepath):
    '''
    Function:
        1.Load the message data from the Database
        2.Generate Feature Dataframes of features from message data, Target Dataframes of multiple category labels and category_names
    
    Args:
        database_filepath (str): filepath of the database
        
    Returns:        
        1.X(DataFrame): Feature Dataframes of features from message data
        2.Y(DataFrame): Target Dataframes of multiple category labels
        3.category_names(list of str): category labels
    '''
    
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterResponse',engine)
    
    X = df['message']
    Y = df[df.columns[4:]]
    category_names = Y.columns.tolist()
    return X,Y,category_names

def tokenize(text):
    """
    Function: split text into words, convert to lowercase and return the root form of the words
    
    Args:
      text(str): the message
      
    Return:
      process_token (list of str): a list of the root form of the message words
    """
    
    # Tokenize text into words
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    # Lemmatization and lowercase
    process_tokens = []
    for tok in tokens:
        process_tok = lemmatizer.lemmatize(tok).lower().strip()
        process_tokens.append(process_tok)

    return process_tokens
    

def build_model():
    """
     Function: build a model for classifing the disaster messages after applying model pipeline and GridSearch to find the optimum model parameters
     
     Args: none
     
     Return:
       cv: an "optimum" classification model after applying 
     """
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf',  MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
        'clf__estimator__n_estimators':[50,100]
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2, n_jobs=-1, verbose=3)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Function: Print the average accuracy of the model and performance matrix of the model for all the different categories based on the classification_report
    
    Args:
        model               : Model is used for the predictions 
        X_test (Data Frame) : Feature Data Frames to be used for predictions
        Y_test (Data Frame) : Target Data Frames which are Values of multiple labels
        category_names (list of str): Name of Target Columns
        
    Returns:
        None
    '''
    
    # Testing the model and Predict the vakue 
    Y_pred = model.predict(X_test)
    
    # Calculate the model's average accuracy
    accuracy = (Y_pred == Y_test.values).mean()
    print('The model average accuracy is {:.3f}'.format(accuracy))
    print('*'*60)
    print('\n')
    
    # Printing the classification report for each label
    idx = -1
    for col in category_names:
        idx = idx + 1
        print('Feature {}: {}'.format(idx, col))
        print(classification_report(Y_test[col], Y_pred[:, idx]))      
    

def save_model(model, model_filepath):
    """
    Function: Save the model as a pickle file
    
    Args:
    model: the classification model
    model_filepath (str): the path of model's pickle file
    
    Returns:
        None   
    """
    
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


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
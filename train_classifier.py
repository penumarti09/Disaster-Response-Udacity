import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
import re
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score, make_scorer, fbeta_score, f1_score
from sklearn.model_selection import GridSearchCV
from sqlalchemy import create_engine
from scipy.stats.mstats import gmean
from sklearn.metrics import classification_report
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.base import BaseEstimator,TransformerMixin
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
import pickle
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def data_load(database_filepath):
    
    '''
      The data_load function is to load data from database located at the path to data base "database_filepath" to the dataframe
      The input for this function would be database_filepath: File path of the SQL database
      The Output would be the following
      category_names: Labels for 36 categories
      X: Message data (features)
      Y: Categories (target)    
    
    '''
    engine = create_engine('sqlite:///data/DisasterResponse.db') #This is for the path for creating engine
    df = pd.read_sql('SELECT * FROM FigureEight', con = engine) #This is for reading the  data frame
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = list(df.columns[4:])
    return X, Y, category_names

def tokenize(text):
    '''
     This function is written to tokenize and clean text
     The Input for this function would be text: original message text
     The Input for this function would be:lemmed: Tokenized, cleaned, and lemmatized text
    
    '''
    found_urls = re.findall(url_regex, text)
    for url in found_urls:
        text = text.replace(url, "urlplaceholder")
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

def build_model():
    '''
     This function is for Building a ML pipeline using ifidf, random forest, and gridsearch
     There would not be Input for this function
     The Output for this cuntion would be Results of GridSearchCV
    
    '''
    pipeline = Pipeline([
                        ('vect', CountVectorizer(tokenizer=tokenize)),
                       ('tfidf', TfidfTransformer()),
                        ('clf', MultiOutputClassifier(RandomForestClassifier()))
                        ])
    parameters = {
                  'clf__estimator__n_estimators': [50, 100],
                 #'clf__estimator__min_samples_split': [2, 3, 4],
                  #'clf__estimator__criterion': ['entropy', 'gini']
                 }
    
    cv = GridSearchCV(pipeline, param_grid=parameters , verbose=3)
    return cv

def fscore_multiop(y_true,y_pred,beta=1):
    '''
    This function is used to generate multi-output fscore values
    The Input for this function are as follows: 
        y_true: True lables for test data
        y_pred: Predicted lables for test data
        beta: beta value is passed as one
    The Output for the function would be :f1score: F1 score value
    '''
    score_list = []
    if isinstance(y_pred, pd.DataFrame) == True:
        y_pred = y_pred.values
    if isinstance(y_true, pd.DataFrame) == True:
        y_true = y_true.values
    for column in range(0,y_true.shape[1]):
        score = fbeta_score(y_true[:,column],y_pred[:,column],beta,average='weighted')
        score_list.append(score)
    f1score_npy = np.asarray(score_list)
    f1score_npy = f1score_npy[f1score_npy<1]
    f1score = gmean(f1score_npy)
    return  f1score

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    This function is to evaluate performance of the model using test data set
    The Input for the function would be: 
        category_names: Labels for 36 categories
        X_test: Test data features
        Y_test: True lables for test data
        model: Model to be evaluated
    The Output for the function would be: Displaying accuracy and classfication report for each category
    
    '''
    Y_pred = model.predict(X_test)

# Calculating accuracy for each of the category names.
    for i in range(len(category_names)):
        print("Category:", category_names[i],"\n", classification_report(Y_test.iloc[:, i].values, Y_pred[:, i]))
        print('Accuracy of %25s: %.2f' %(category_names[i], accuracy_score(Y_test.iloc[:, i].values, Y_pred[:,i])))
#Getting overall accuracy and fscore
        multiple_f1score = fscore_multiop(Y_test,Y_pred, beta = 1)
        overall_accuracy = (Y_pred == Y_test).mean().mean()
        print('Average overall accuracy {0:.2f}% \n'.format(overall_accuracy*100))
        print('F1 score (custom definition) {0:.2f}%\n'.format(multiple_f1score*100))


def save_model(model, model_filepath):
 	
    '''
    Save model as a pickle file : saving the model to disk
    The Input for this function would be: 
        model_filepath: path of the output pick file
        model: Model to be saved        
    The Output for this function would be: A pickle file of saved model
    
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = data_load(database_filepath)
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
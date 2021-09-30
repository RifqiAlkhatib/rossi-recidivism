from logs import logDecorator as lD 
import jsonref
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV


config = jsonref.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.modelling.modelling'
data_config = jsonref.load(open('../config/data-config.json'))
clean_data = data_config['inputs']['clean_data']
target = data_config['params']['target']


@lD.log(logBase + '.make_x_y')
def make_x_y(logger, data=clean_data, target=target):
    '''Creates scaled X and y data for modelling
    
    Args:
        logger : {logging.Logger}
            The logger used for logging error information
        data: {path to csv}
            csv file containing cleaned data
        target: {str}
            target column name
    '''

    df = pd.read_csv(data)
    X = df.drop(columns=target)
    ss = StandardScaler()
    X_sc = ss.fit_transform(X)
    y = df[target]
    
    return X_sc, y

@lD.log(logBase + '.logreg')
def logreg(logger, X, y):
    '''Fits X and y data to a logistic regression model and saves the model

    Args:
        logger : {logging.Logger}
            The logger used for logging error information
        X: {dataframe}
            dataframe containing X variables
        y: {series}
            series containing y values
    '''

    lr_params = data_config['params']['logreg_params']
    lr = LogisticRegressionCV(
        Cs = lr_params['Cs'],
        n_jobs = lr_params['n_jobs'],
        max_iter = lr_params['max_iter']
    )
    lr.fit(X, y)
    print('Logistic Regression model fit done')

    path = data_config['outputs']['logreg_model']
    pickle.dump(lr, open(path, 'wb'))
    print(f'Model saved to {path}')

    return

@lD.log(logBase + '.main')
def main(logger, resultsDict):
    '''main function for module1
    
    This function finishes all the tasks for the
    main function. This is a way in which a 
    particular module is going to be executed. 
    
    Args:
        logger : {logging.Logger}
            The logger used for logging error information
    '''

    X, y = make_x_y()
    print('make_x_y done')

    logreg(X, y)
    print('logreg done')

    return
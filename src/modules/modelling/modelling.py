from logs import logDecorator as lD 
import jsonref
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split


config = jsonref.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.modelling.modelling'
data_config = jsonref.load(open('../config/data-config.json'))
clean_data = data_config['inputs']['clean_data']
target = data_config['params']['target']


@lD.log(logBase + '.make_x_y')
def make_x_y(logger, data=clean_data, target=target):
    '''Creates scaled X and y train & test data for modelling
    
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
    y = df[target]

    tts_params = data_config['params']['tts_params']
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=tts_params['test_size'],
        random_state=tts_params['random_state']
    )

    ss = StandardScaler()
    X_train_sc = ss.fit_transform(X_train)
    X_test_sc = ss.transform(X_test)
    
    return X_train_sc, X_test_sc, y_train, y_test


@lD.log(logBase + '.logreg')
def logreg(logger, X, y):
    '''Fits X and y data to a logistic regression model and saves the model

    Args:
        logger : {logging.Logger}
            The logger used for logging error information
        X: {array}
            array containing scaled X variables
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

    return lr


@lD.log(logBase + '.coefs')
def coefs(logger, model, data=clean_data, target=target):
    """Generates table of coefficients for each X variable of the model

    Parameters
    ----------
    logger : {logging.logger}
       The logger used for logging error information
    model : {model}
        Model being assessed
    data: {path to csv}
        csv file containing cleaned data
    target: {str}
        target column name
    """

    df = pd.read_csv(data)
    X = df.drop(columns=target)
    coef_df = pd.DataFrame({'Variable': X.columns, 'Coefficient': model.coef_[0]})\
        .sort_values(by='Coefficient')

    coef_path = data_config['outputs']['logreg_coefs']
    coef_df.to_csv(coef_path, index=False)
    print(f'Coefficients saved to {coef_path}')

    return


@lD.log(logBase + '.metrics')
def metrics(logger, model, X_train, y_train, X_test, y_test):
    """Generates metrics to assess performance of model

    Parameters
    ----------
    logger : {logging.Logger}
        The logger used for logging error information
    model : {model}
        Model being assessed
    X_train : {array}
        Training data (X variables)
    y_train : {series}
        Training data (y target variable)
    X_test : {array}
        Testing data which model has not been exposed to (X variables)
    y_test : {series}
        Testing data which model has not been exposed to (y target variable)
    """

    results = dict()
    y_preds = model.predict(X_test)
    results['Train Accuracy'] = model.score(X_train, y_train)
    results['Test Accuracy'] = accuracy_score(y_test, y_preds)
    results['Precision'] = precision_score(y_test, y_preds)
    results['Recall'] = recall_score(y_test, y_preds)

    metric_cols = data_config['params']['metrics_cols']
    res_df = pd.DataFrame(results.items(), columns=metric_cols)

    metrics_path = data_config['outputs']['logreg_metrics']
    res_df.to_csv(metrics_path, index=False)
    print(f'Metrics saved to {metrics_path}')

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

    X_train, X_test, y_train, y_test = make_x_y()
    print('make_x_y done')

    lr = logreg(X_train, y_train)
    print('logreg done')

    coefs(lr)
    print('coefficients generated')

    metrics(lr, X_train, y_train, X_test, y_test)
    print('metrics generated')

    return
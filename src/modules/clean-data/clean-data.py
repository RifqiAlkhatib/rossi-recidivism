from logs import logDecorator as lD 
import jsonref
import pandas as pd


config = jsonref.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.clean-data.clean-data'
data_config = jsonref.load(open('../config/data-config.json'))
raw_data = data_config['inputs']['raw_data']
clean_data = data_config['inputs']['clean_data']


@lD.log(logBase + '.drop_cols')
def drop_cols(logger, data=raw_data):
    '''Drops unnecessary columns for modelling
    
    Args:
        logger : {logging.Logger}
            The logger used for logging error information
        data: {path to csv}
            location containing csv file containing convict data
    '''

    df = pd.read_csv(data)
    cols_to_drop = data_config['params']['drop_cols']
    df.drop(columns=cols_to_drop, inplace=True)

    return df

@lD.log(logBase + '.save_data')
def save_data(logger, df, data=clean_data):
    '''Exports cleaned dataframe to data folder

    Args:
        logger : {logging.Logger}
            The logger used for logging error information
        df: {dataframe}
            dataframe containing cleaned data
        data: {path to csv}
            location containing csv file containing cleaned data
    '''

    df.to_csv(data, index=False)
    print(f'Cleaned df saved to {data}')
    
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

    clean_df = drop_cols()
    print('drop_cols done')

    save_data(clean_df)
    print('save_data done')

    return


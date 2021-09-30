from logs import logDecorator as lD 
import jsonref
import pandas as pd
import os
from lib.databaseIO import pgIO

config = jsonref.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.pgIOtest.pgIOtest'

schema = "r21r1_dcdm"
table = "person"
columns = ["person_id", "marital_status"]

@lD.log(logBase + '.test1')
def test1(logger):
    """
    """

    query = """
        SELECT person_id, marital_status
        FROM r21r1_dcdm.person"""

    df = pd.DataFrame(pgIO.getAllData(query), columns=['person_id', 'marital_status'])

    return df



@lD.log(logBase + '.main')
def main(logger, resultsDict):
    '''main function for module1
    
    This function finishes all the tasks for the
    main function. This is a way in which a 
    particular module is going to be executed. 
    
    Parameters
    ----------
    logger : {logging.Logger}
        The logger used for logging error information
    resultsDict: {dict}
        A dintionary containing information about the 
        command line arguments. These can be used for
        overwriting command line arguments as needed.
    '''

    df = test1()
    print("test1 done")
    print(df)

    return


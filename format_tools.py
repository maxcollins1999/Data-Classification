### Preamble ###################################################################
#
# Author            : Max Collins
#
# Github            : https://github.com/maxcollins1999
#
# Title             : format_tools.py
#
# Date Last Modified: 27/9/2020
#
# Notes             : A raw data formatter seperate to sklearn pipelines so as 
#                     to preserve column names and row removal capabilities.
#
################################################################################

### Imports ####################################################################

#Global
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

################################################################################

class raw_formatter():
    """Class to format data from file. Does not inherit from sklearn 
    for pipeline development because columns must be preserved and 
    rows are removed.
    """

    def __init__(self):
        self.cat_cols = None
        self.num_cols = None
        self.num_imputer = None
        self.cat_imputer = None 
        self.encoder = None


    def fit_transform(self, Xy_data, y_col, num_cols, cat_cols):
        """Takes a pandas dataframe object, indexer of the response column,
        a list of the numeric columns and categorical columns. Returns 
        transformed dataset.
        """

        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.y_col = y_col
        Xy = Xy_data.copy()
        #Formatting data types
        Xy = self.__format_dtypes(Xy)
        #Imputing
        self.num_imputer = SimpleImputer(missing_values=np.nan, strategy='mean').fit(Xy[num_cols])
        self.cat_imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='Unknown').fit(Xy[cat_cols])
        Xy = self.__impute_data(Xy)
        #Removing duplicate data points
        Xy = Xy.drop_duplicates()
        #Encoding
        self.encoder = OneHotEncoder().fit(Xy[self.cat_cols].astype(str))
        Xy = self.__encode(Xy)

        return Xy


    def transform(self, Xy):
        """Takes a pandas dataframe and transforms the data, returning 
        the transformed dataframe.
        """

        Xy = Xy.copy()

        #Formatting data types
        Xy = self.__format_dtypes(Xy)
        #Imputing
        Xy = self.__impute_data(Xy)
        #Encoding
        Xy = self.__encode(Xy)

        return Xy


    def __format_dtypes(self, Xy):
        """Formats the datatypes converting numerics to np.number 
        and categorical to object.
        """

        for col in self.num_cols:
            Xy[col] = Xy[col].astype(np.number)
        for col in self.cat_cols:
            Xy[col] = Xy[col].astype(object)
        for col in Xy:
            if (col not in self.num_cols) and (col not in self.cat_cols) and \
            col != self.y_col:
                Xy = Xy.drop(col, axis=1)
        return Xy


    def __impute_data(self, Xy):
        """Imputes the data
        """

        Xy[self.num_cols] = self.num_imputer.transform(Xy[self.num_cols])
        Xy[self.cat_cols] = self.cat_imputer.transform(Xy[self.cat_cols])
        return Xy


    def __encode(self, Xy):
        """Encodes the data
        """

        Xy = Xy.reset_index(drop=True)
        Xy = Xy.drop(self.cat_cols,axis=1).join(pd.DataFrame(self.encoder.transform(Xy[self.cat_cols].astype(str)).toarray()))
        return Xy

### Private Methods ############################################################



################################################################################

### Tests ######################################################################



################################################################################
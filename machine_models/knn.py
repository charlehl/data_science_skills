import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler

class knnCustom(object):
    """
        Custom implementation of k-nearest neighbor.  Can calculate distance based on either
        euclidean or manhattan distance.  Distance can be easily modified to support other distance 
        calculations supported by numpy.linalg.norm.
        
        Missing data is allowed and is handled during the kneighbor calculation by ignoring those data points
        in distance calculation.
    """
    def __init__(self, k=5):
        """
            Inputs:
                k - set for the requested number of nearest neighbors
                    
            Notes:
                If distance of neighbor has ties, will return all ties
        """
        # Initialize object
        self.k = k
        self.scaler = None
        self.data = None
        self.columnSetList = None
    def fit(self, X, scaleFlag=False):
        """
            Used for fitting training data.  Basically just allowing ability to scale data.
            
            Inputs:
                X - training data for KNN
                scaleFlag - boolean opeartor determines if training data will be scaled by MinMaxScaler
        """
        # Create a mask for each training row denoting data that is not NaN
        self.columnSetList = []
        rowList = X.values.tolist()
        # Create column mask to filter NaNs
        for row in rowList:
            # Find all indexes in row that are not NaN
            nonNaNs = [colMask[0] for colMask in np.argwhere(~np.isnan(row))]
            # Convert to set to make intersection calculation easier
            self.columnSetList.append(set(nonNaNs))
        # Scale training data
        if scaleFlag == True:
            self.scaler = MinMaxScaler()
            self.data = self.scaler.fit_transform(X)
        else:
            self.scaler = None
            self.data = X.values.tolist()
    def kneighbors(self, X, metric="euclidean"):
        """
            Predict the k nearest neighbors for each X data
            Inputs:
                X - Data to find nearest neighbors to training data.
                metric - specify distance metric to use for KNN.  Currently supports
                         only euclidean and manhattan distance.
            Outputs:
                Returns a dictionary with indexes to neighbors and distances to neighbors
                {
                 "indexes": [...],
                 "distances": [...]
                }
                Ordered by descending distance respectively.
            Notes: 
                If data in training or test is NaN then the algorithm will ignore those in the distance calculation.
        """
        # Select distance metric for neighbors
        if metric == "euclidean":
            p = 2
        elif metric == 'manhattan':
            p = 1
        else:
            raise ValueError("metric supported is only 'euclidean' or 'manhattan'")

        # If training data was scaled then scale testing data accordingly
        if self.scaler != None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values.tolist()
        # Array to hold distance calculations for each row of testing data
        distances = []
        # Array to hold the indexes of the KNN neighbor training data
        indexes = []
        # Process each row for the k-nearest neighbor
        for test in X_scaled:
            # Create column mask for test data
            colMaskSetTest = set([colMask[0] for colMask in np.argwhere(~np.isnan(test))])
            # List to hold the distance calculations for each neighbor
            dist_calcs = []
            # Calculate the distance
            for idx, row in enumerate(self.data):
                # Create intersection of columns with non NaNs
                index_list = list(colMaskSetTest  & self.columnSetList[idx])
                # Pull data from row
                v1 = list(map(row.__getitem__, index_list))
                v2 = list(map(test.__getitem__, index_list))
                # Calculate the distance for test
                dist_calcs.append(np.linalg.norm(np.subtract(v1, v2), p))
            # Create a dataframe to get sorted list
            df = pd.DataFrame(dist_calcs, columns=["distances"])
            df = df.sort_values(by="distances")
            # Calculate the target distance
            target_dist = df.iloc[self.k-1,0]
            # Return all data at that distance or less
            df = df[df["distances"] <= target_dist]
            # Append to lists
            distances.append(df["distances"].tolist())
            indexes.append(list(df.index))
        # Return all of the knn neighbors for distances and indexes
        return {"indexes": indexes, "distances": distances}

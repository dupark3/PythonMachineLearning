import numpy as np
from sklearn import preprocessing

data = np.array([[3, -1.5, 2, -5.4], [0, 4, -0.3, 2.1], [1, 3.3, -1.9, -4.3]])
print("Data = ", data)

# remove mean to center on zero, allows different data sets to be comparable
# for example, 
data_standardized = preprocessing.scale(data)
print("Mean = ", data_standardized.mean(axis=0))
print("Std Deviation = ", data_standardized.std(axis=0))

data_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
data_scaled = data_scaler.fit_transform(data)
print("Min Max Scaled data = \n", data_scaled)

data_normalized = preprocessing.normalize(data, norm='l1')
print("L1 normalized data = \n", data_normalized)
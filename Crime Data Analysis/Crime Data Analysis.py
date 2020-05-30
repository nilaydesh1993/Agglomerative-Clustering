"""
Created on Mon Apr 20 14:36:23 2020
@author: DESHMUKH
HIERARCHICAL CLUSTERING
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

from sklearn.cluster import AgglomerativeClustering

# ================================================================================================
# Business Problem :- Perform Clustering for the crime data and identify the number of clusters.
# ================================================================================================

crime = pd.read_csv("crime_data.csv")
crime = crime.rename({ 'Unnamed: 0' : 'city'},axis = 1)
crime.head()
crime.info()
crime.isnull().sum()
crime.shape
crime.columns

# Summary
crime.describe()

# Histogram
crime.hist()

# Scatter Plot
sns.pairplot(crime,diag_kind="kde")

# Standardization of Data (We can also use normalization)
from sklearn import preprocessing
crime_std = preprocessing.scale(crime.iloc[:,1:5])
crime_std = pd.DataFrame(crime_std,columns =['Murder', 'Assault', 'UrbanPop', 'Rape'] )

# Creating linkage for Dendogrma
L = sch.linkage(crime_std, method = "complete", metric = "euclidean" )

# Dendogram
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    L,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=10.,  # font size for the x axis labels
)
plt.show()

# Now applying AgglomerativeClustering choosing 4 as clusters from the dendrogram.
h_complete = AgglomerativeClustering(n_clusters=4, linkage='complete',affinity = "euclidean").fit(crime_std) 

# Converting result into Dataframe or Series
cluster_labels = pd.DataFrame((h_complete.labels_),columns = ['cluster'])

# Concating lable dataframe into original data frame
crime_final = pd.concat([cluster_labels,crime],axis=1)

# Getting aggregate mean of each cluster
crime_final.iloc[:,2:].groupby(crime_final.cluster).mean()

# Creating a csv file 
#crime_final.to_csv("crime_final.csv",encoding="utf-8")

                #---------------------------------------------------------#







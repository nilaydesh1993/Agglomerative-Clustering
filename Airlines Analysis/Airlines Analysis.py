"""
Created on Sun Apr 19 19:41:27 2020
@author: DESHMUKH
HIERARCHICAL CLUSTERING 
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch

from sklearn.cluster import AgglomerativeClustering 
pd.set_option('display.max_column',None)

# ===================================================================================
# Business Problem :- Obtain optimum number of clusters for the airlines data. 
# ===================================================================================

airline1 = pd.read_excel("EastWestAirlines.xlsx",sheet_name = 'data')
airline = airline1
airline = airline.drop('ID#',axis = 1)
airline.head()
airline.columns
airline.isnull().sum()
airline.info()
airline.shape

# Summary 
airline.describe()

# Histogram
airline.hist()

# Scatter Plot
sns.pairplot(airline)

# Normalization of Data instead of std.(beacuse it contain Binary value)
from sklearn.preprocessing import normalize
airline.iloc[:,0:10] = normalize(airline.iloc[:,0:10])

# Creating linkage for Dendrogram
z = sch.linkage(airline, method="complete",metric="euclidean")

# Dendrogram
plt.figure(figsize=(50, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=2.,  # font size for the x axis labels
)
plt.show()

# Now applying AgglomerativeClustering choosing 5 as clusters from the dendrogram
h_complete = AgglomerativeClustering(n_clusters=5, linkage='complete',affinity = "euclidean").fit(airline) 

# Converting result into Dataframe or Series
cluster_labels = pd.DataFrame((h_complete.labels_),columns = ['cluster'])

# Concating lable dataframe into original data frame
airlinefinal = pd.concat([cluster_labels,airline1],axis=1)

# Getting aggregate mean of each cluster
airlinefinal.iloc[:,2:].groupby(airlinefinal.cluster).mean()

# Creating a csv file 
#airlinefinal.to_csv("Airlinefinal.csv",encoding="utf-8")

                #-------------------------------------------------#

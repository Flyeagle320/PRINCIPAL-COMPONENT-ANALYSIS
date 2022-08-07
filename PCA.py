# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 05:55:45 2022

@author: Rakesh
"""

###Please note this assignment has been finished based on few coding reference from Github ##

###problem Statement 1###########################

#Importing packages #
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale #normalization#
import seaborn as sns
from scipy.cluster.hierarchy import linkage #linkage for clustering
import scipy.cluster.hierarchy as sch #hiearchichal clustering#
from sklearn.cluster import KMeans ##Kmeans clustering ##

##importing data##
wine_data = pd.read_csv('D:/DATA SCIENCE ASSIGNMENT/Datasets_PCA/wine.csv')

wine_data.head()
wine_data.describe()
##lets drop Type columns in data set as it doenst add value ##

new_wine_date2 = wine_data.drop(['Type'], axis=1)
##checking na and null value##
new_wine_date2.isnull().sum()
new_wine_date2.isna().sum()
#checking duplicate value##
dup1 = new_wine_date2.duplicated()
sum(dup1)

##scaling data#
new_wine_date_norm = scale(new_wine_date2)

from sklearn.decomposition import PCA
##PCA value calculation#
pca= PCA(n_components=6)

new_wine_date_pca= pca.fit_transform(new_wine_date_norm)

var = pca.explained_variance_ratio_ ##getting variance in data#

pca.components_ ###view pca value

cumsum_var = np.cumsum(np.round(var,decimals=4)*100) ##getting cumulative variance in percentage

plt.plot(cumsum_var,color = 'blue') ##plotting pca variance#

new_wine_date_pca = pd.DataFrame(new_wine_date_pca)

new_wine_date_pca.columns = 'comp0', 'comp1' , 'comp2' , 'comp3' , 'comp4' , 'comp5' ##namimg pca columns

new_wine_date_pca_final =  pd.concat([wine_data.Type,new_wine_date_pca.iloc[:,0:3]], axis = 1)

## Boxplottimg to check outlier##

sns.boxplot(new_wine_date_pca_final.comp0);plt.title('Boxplot');plt.show()
sns.boxplot(new_wine_date_pca_final.comp1);plt.title('Boxplot');plt.show()
sns.boxplot(new_wine_date_pca_final.comp2);plt.title('Boxplot');plt.show()

##there are outlier in comp2
IQR =  new_wine_date_pca_final['comp2'].quantile(0.75)- new_wine_date_pca_final['comp2'].quantile(0.25)
lower_limit_comp2 = new_wine_date_pca_final['comp2'].quantile(0.25)-(IQR*1.5)
higher_limit_comp2 = new_wine_date_pca_final['comp2'].quantile(0.75)+(IQR*1.5)
new_wine_date_pca_final['comp2']=pd.DataFrame(np.where(new_wine_date_pca_final['comp2']>higher_limit_comp2,higher_limit_comp2,
                                                       np.where(new_wine_date_pca_final['comp2']<lower_limit_comp2,lower_limit_comp2,new_wine_date_pca_final['comp2'])))
sns.boxplot(new_wine_date_pca_final.comp2);plt.title('Boxplot');plt.show()

##hiearchical clustering##

linkage_complete = linkage(new_wine_date_pca_final, method = 'complete', metric='euclidean')
plt.figure(figsize=(15,8));plt.title('Hiearchical clustering for complete linkage');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(linkage_complete,leaf_rotation=0, leaf_font_size=10)
plt.show()

##kmeans ##
##gettting cluster value for cluster range froom 2-9
TWSS = []
k = list(range(2,9))
for i in k:
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(new_wine_date_pca_final)
    TWSS.append(kmeans.inertia_)

TWSS
##scree plotting##
plt.plot(k,TWSS, 'ro-');plt.xlabel('No_of_cluster');plt.ylabel('total_within_SS')

##model building ##
model_new_wine_date_pca_final= KMeans(n_clusters=3)
model_new_wine_date_pca_final.fit(new_wine_date_pca_final)

##storing clustering in original datafram##
model_new_wine_date_pca_final.labels_ ##assigning label to each rows# 
cluster_new_wine_date_pca_final= pd.DataFrame(model_new_wine_date_pca_final.labels_) ##converting numpy in panda series object#
new_wine_date_pca_final['cluster']= cluster_new_wine_date_pca_final

##replacing index of column##
new_wine_date_pca_final =new_wine_date_pca_final.iloc[: , [4,0,1,2,3]]

#renaming original cluster data to match with calculated cluster data
#here the given clustered data is almost similar to the calcluated cluster data found out using pca values
new_wine_date_pca_final['Type'].replace({1:0, 2:2, 3:1},inplace = True)

new_wine_date_pca_final.to_csv('new_wine_data', encoding ='utf-8')

import os
os.getcwd()
##########################################Problem 5###############################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale #normalization#
import seaborn as sns
from scipy.cluster.hierarchy import linkage #linkage for clustering
import scipy.cluster.hierarchy as sch #hiearchichal clustering#
from sklearn.cluster import KMeans ##Kmeans clustering ##

##loading dataset##
heart= pd.read_csv('D:/DATA SCIENCE ASSIGNMENT/Datasets_PCA/heart disease.csv')

##lets drop target column as it doesnt add value ##
heart_new = heart.drop(['target'], axis= 1)

heart_new.describe()
##cheking na & null value##
heart_new.isna().sum()
heart_new.isnull().sum()

#checking duplicated value ##
dup1 = heart_new.duplicated()
sum(dup1)

heart_new = heart_new.drop_duplicates()

##scaling data##

heart_new_norm = scale(heart_new)

##calculating pca values##
pca = PCA(n_components=6)
heart_new_pca =pca.fit_transform(heart_new_norm)

##calculating variance for PCA value ##
var =pca.explained_variance_ratio_

##viewing pca values
pca.components_

##calculating cumulative variance of the PCA values in percentages##

cumsum_var= np.cumsum(np.round(var,decimals=4)*100)

##plotting the variance data##
plt.plot(cumsum_var, color ='blue')

heart_new_pca = pd.DataFrame(heart_new_pca)
heart_new_pca.columns = 'comp0' , 'comp1' , 'comp2' , 'comp3' ,'comp4' , 'comp5'

##lets take first 3 component ##
heart_new_pca_final = heart_new_pca.iloc[: , 0:3]

##boxlotting for outlier analysis#

sns.boxplot(heart_new_pca_final.comp0);plt.title('Boxplot');plt.show()
sns.boxplot(heart_new_pca_final.comp1);plt.title('Boxplot');plt.show()
sns.boxplot(heart_new_pca_final.comp2);plt.title('Boxplot');plt.show()

##removing outlier using IQR#

IQR =  heart_new_pca_final['comp1'].quantile(0.75)- heart_new_pca_final['comp1'].quantile(0.25)
lower_limit_comp1 = heart_new_pca_final['comp1'].quantile(0.25)-(IQR*1.5)
higher_limit_comp1 = heart_new_pca_final['comp1'].quantile(0.75)+(IQR*1.5)
heart_new_pca_final['comp1']=pd.DataFrame(np.where(heart_new_pca_final['comp1']>higher_limit_comp1,higher_limit_comp1,
                                                       np.where(heart_new_pca_final['comp1']<lower_limit_comp1,lower_limit_comp1,heart_new_pca_final['comp1'])))
sns.boxplot(heart_new_pca_final.comp1);plt.title('Boxplot');plt.show()

IQR =  heart_new_pca_final['comp2'].quantile(0.75)- heart_new_pca_final['comp2'].quantile(0.25)
lower_limit_comp2 = heart_new_pca_final['comp2'].quantile(0.25)-(IQR*1.5)
higher_limit_comp2 = heart_new_pca_final['comp2'].quantile(0.75)+(IQR*1.5)
heart_new_pca_final['comp2']=pd.DataFrame(np.where(heart_new_pca_final['comp2']>higher_limit_comp2,higher_limit_comp2,
                                                       np.where(heart_new_pca_final['comp2']<lower_limit_comp2,lower_limit_comp2,heart_new_pca_final['comp2'])))
sns.boxplot(heart_new_pca_final.comp2);plt.title('Boxplot');plt.show()


##linkage for hiearchical cluster and dendrgram##
linkage_complete =linkage(heart_new_pca_final ,method='complete', metric='euclidean')
plt.figure(figsize=(15,8)); plt.title('Hiearchical clustering dendrogram for complete linkage'); plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(linkage_complete , leaf_rotation=0 , leaf_font_size=10)
plt.show()

##kmeand model building##
TWSS = []
k = list(range(2,9))
for i in k:
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(heart_new_pca_final)
    TWSS.append(kmeans.inertia_)
TWSS

##scree plotting ##

plt.plot(k, TWSS , 'ro-');plt.xlabel('No_of_clusters');plt.ylabel('total_within_SS')

##model building ##
model_heart_new_pca_final= KMeans(n_clusters=3)
model_heart_new_pca_final.fit(heart_new_pca_final)
model_heart_new_pca_final.labels_ #getting labels asigned to each row
cluster_heart_new_pca_final =pd.Series(model_heart_new_pca_final.labels_) ##converting numpy into panda series#
heart_new_pca_final['cluster'] = cluster_heart_new_pca_final
heart_new_pca_final =heart_new_pca_final.iloc[: , [3,0,1,2]]

##since the original cluster is given in only 2 cluster in statement , so let calculate Kmeans for 2 cluster#

model_heart_new_pca_final= KMeans(n_clusters=2)
model_heart_new_pca_final.fit(heart_new_pca_final)
model_heart_new_pca_final.labels_ #getting labels asigned to each row
cluster_heart_new_pca_final =pd.Series(model_heart_new_pca_final.labels_) ##converting numpy into panda series#
heart_new_pca_final['cluster'] = cluster_heart_new_pca_final
heart_new_pca_final =heart_new_pca_final.iloc[: , [3,0,1,2]]

heart_new_pca_final1= pd.concat([heart.target , heart_new_pca_final.iloc[: , 0:4]], axis=1)
heart_new_pca_final1['target'].replace({1:0 , 0:1}, inplace= True)
#comparison is done and found out that the given cluster[target] in question is different to cluster values for pca components








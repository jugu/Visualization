import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets, preprocessing, manifold
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
import math
import pandas as pd
class DataAnalyzer(object):
    def __init__(self, folder='', file="Batting.csv", delimiter=","):
        self.rootFolder = folder
        self.filename = file
        self.delimiter = delimiter

    def readDataFromFile(self):
        dataframe = pd.read_csv(self.rootFolder+self.filename,
                                usecols=['playerID', 'yearID', 'stint', 'teamID', 'G', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'IBB', 'HBP', 'SH', 'SF', 'GIDP'])
        dataframe.fillna(0, inplace=True)
        return dataframe
        #self.data = np.genfromtxt(self.rootFolder+'\\'+filename, delimiter=delimiter, skiprows=1, usecols=(3, -1), filling_values=0, missing_values=0)

    def performKMeansOnline(self, data, start, end):
        np.random.seed(5)
         #return dataframe.loc[:,'G', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'IBB', 'HBP', 'SH', 'SF', 'GIDP'].values
        data = data.drop(['playerID', 'yearID', 'stint', 'teamID'], axis=1)
        data_normalized = preprocessing.normalize(data, norm='l2')
        clusterdata = {}
        graphElbowCluster = []
        clusterlabel = []
        clusterdata['data'] = graphElbowCluster
        clusterdata['labels'] = clusterlabel
        for i in range(start, end):
            estimator = KMeans(n_clusters=i)
            #estimator = MiniBatchKMeans(n_clusters=8,init='k-means++',max_iter=300,batch_size=100)
            estimator.fit(data_normalized)
            labels = estimator.labels_
            for k in range(i):
                clusterlabel.append({"label" : np.argwhere(labels==k)})
            graphElbowCluster.append({"x":i,"y":estimator.inertia_})
        return clusterdata

    def doRandomSampling(self, data, clusterdata, limit=100):
        newdata = data.sample(n=limit)
        dataobj = {}
        #labels = {}
        labels = []
        length = len(clusterdata['labels'])
        #for i in range(length):
        #    labels[str(i)] = []
        dataobj['df'] = newdata
        dataobj['cluster'] = labels
        for row in newdata.iterrows():
            for i in range(length):
                if row[0] in clusterdata['labels'][i]['label']:
                    #labels[str(i)].append(row[0])
                    labels.append(i)
                    break
        #print labels
        #self.doAdaptiveSampling(data, clusterdata, limit)
        return dataobj

    def doAdaptiveSampling(self, data, clusterdata, limit=100):
        length = len(clusterdata['labels'])
        counts = []
        for i in range(length):
            counts.append(clusterdata['labels'][i]['label'].size)
        norm = [int(round((float(i)/sum(counts))*100)) for i in counts]
        indexArray = []
        labels = []
        #labels = {}
        for i in range(len(norm)):
            np.random.shuffle(clusterdata['labels'][i]['label'])
            indexArray = np.append(indexArray, clusterdata['labels'][i]['label'][0:norm[i]])
            #labels[str(i)] = clusterdata['labels'][i]['label'][0:norm[i]].tolist()
            labels = labels + [i for x in range(norm[i])]
        print labels
        adaptiveDF = data.ix[indexArray]
        retdata = {}
        retdata['df'] = adaptiveDF
        retdata['cluster'] = labels
        return retdata

    def doIsomap(self, data, labels):
        mdata = data.drop(['playerID', 'yearID', 'stint', 'teamID'], axis=1)
        mdata = manifold.Isomap(n_neighbors=5, n_components=2).fit_transform(mdata)
        cluster = 0
        isomapdata=[]
        for row in mdata:
            rowdata = {}
            rowdata['pointname'] = data.iloc[cluster][0]+','+str(data.iloc[cluster][3])
            rowdata['xvalue'] = row[0]
            rowdata['yvalue'] = row[1]
            rowdata['cluster'] = labels[cluster]
            isomapdata.append(rowdata)
            cluster +=1
        return isomapdata

    def doMDS(self, data, type, labels):
        mdata = data.drop(['playerID', 'yearID', 'stint', 'teamID'], axis=1)
        #data = preprocessing.normalize(data, norm='l2')
        if (type == 'EUCLID'):
            distances =  pairwise_distances(X = mdata, metric = "euclidean")
        elif (type == 'COSINE'):
            distances =  pairwise_distances(X = mdata, metric = "cosine")
        else:
            distances =  pairwise_distances(X = mdata, metric = "correlation")
        mds = manifold.MDS(n_components=2, dissimilarity='precomputed')
        newdata = mds.fit_transform(distances)
        cluster = 0
        mdsdata = []
        for row in newdata:
            rowdata = {}
            rowdata['pointname'] = data.iloc[cluster][0]+','+str(data.iloc[cluster][3])
            rowdata['xvalue'] = row[0]
            rowdata['yvalue'] = row[1]
            rowdata['cluster'] = labels[cluster]
            mdsdata.append(rowdata)
            cluster +=1
        return mdsdata

    def scree(self, data):
        screedata = []
        pca = PCA(n_components=10)
        data_normalized = preprocessing.normalize(data, norm='l2')
        pca.fit(data_normalized)
        count = 1
        for i in pca.explained_variance_:
            rowdata = {}
            rowdata['x'] = count
            rowdata['y'] = (i)
            count += 1
            screedata.append(rowdata)
        return screedata

    def doPCA(self, data, labels):
        mdata = data.drop(['playerID', 'yearID', 'stint', 'teamID'], axis=1)
        screedata = self.scree(mdata);
        pca = PCA(n_components=2)
        pca.fit(mdata.values)
        first_pc = pca.components_[0]
        second_pc = pca.components_[1]
        print pca.explained_variance_ratio_
        print pca.explained_variance_
        transformed_data = pca.transform(mdata)
        cluster = 0
        pcadata = []
        for row in transformed_data:
            rowdata = {}
            rowdata['pointname'] = data.iloc[cluster][0]+','+str(data.iloc[cluster][3])
            rowdata['xvalue'] = row[0]
            rowdata['yvalue'] = row[1]
            rowdata['cluster'] = labels[cluster]
            pcadata.append(rowdata)
            cluster +=1
        return pcadata, screedata

if __name__== "__main__":
    da = DataAnalyzer("")
    da.readDataFromFile()

from flask import Flask, request, jsonify
from flask import render_template

from businesslogic import DataAnalyzer
import json

app = Flask(__name__)
folder = ''
file = "Batting.csv"
delimiter = ","
RANDOMSAMPLING = 0
ADAPTIVESAMPLING = 1
PCA = 0
MDS_EUCLIDEAN = 1
MDS_COSINE = 2
MDS_CORRELATION = 3
ISOMAP = 4

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/baseball/batting/generatecluster")
def generateK():
    clusters = int(request.args.get('clusters'))
    sampling = int(request.args.get('sampling'))
    viz = int(request.args.get('viz'))
    da = DataAnalyzer(folder, file, delimiter)
    data = da.readDataFromFile()
    clusterdata = da.performKMeansOnline(data, 3, 5)
    return jsonify(data = clusterdata['data'])

@app.route("/baseball/batting/visualize")
def visualizeData():
    clusters = int(request.args.get('clusters','3',type=str))
    sampling = int(request.args.get('sampling','0',type=str))
    viz = int(request.args.get('viz','0',type=str))
    print clusters, sampling, viz
    da = DataAnalyzer(folder, file, delimiter)
    data = da.readDataFromFile()
    clusterdata = da.performKMeansOnline(data, clusters, clusters+1)
    if sampling == RANDOMSAMPLING:
        dataobj = da.doRandomSampling(data, clusterdata)
    elif sampling == ADAPTIVESAMPLING:
        dataobj = da.doAdaptiveSampling(data, clusterdata)
    screedata=[]
    if viz == PCA:
        data, screedata = da.doPCA(dataobj['df'],dataobj['cluster'])
    elif viz == ISOMAP:
        data = da.doIsomap(dataobj['df'],dataobj['cluster'])
    else:
        if viz == MDS_EUCLIDEAN:
            data = da.doMDS(dataobj['df'], "EUCLID", dataobj['cluster'])
        elif viz == MDS_COSINE:
            data = da.doMDS(dataobj['df'], "COSINE", dataobj['cluster'])
        else:
            data = da.doMDS(dataobj['df'], "CORRELATION", dataobj['cluster'])
    return jsonify(data = data, scree = screedata)

from textanalysis import *
@app.route("/textanalysis")
def analyzeText():
    valuelist, labellist = tf_idf_genrelines()
    #valuelist, labellist = tf_idf_test()
    coords = datacoordinates(valuelist, labellist)
    return jsonify(data = coords)
#visualizeData()
if __name__ == "__main__":
   app.run(host='0.0.0.0',port=5000,debug=True)
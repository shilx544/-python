from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import urllib.parse
import urllib.request
import json
from time import sleep

def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat

#calculate the distance
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA-vecB, 2)))

def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
    return centroids

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2))) #index of the cluster and the error
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist ** 2
        print(centroids)
        for cent in range(k):
            a = nonzero(clusterAssment[:, 0].A == cent) #nonzero return a 2-D ndarray
            b = dataSet[2]
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment

def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0] #Initialy create one cluster
    for j in range(m):
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j, :]) ** 2
    while (len(centList) < k):
        lowestSSE = inf #sum of squared error
        for i in range(len(centList)): #Try splitting every cluster
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:, 1])
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print("sseSplit, and notSplit: ", sseSplit, sseNotSplit)
            if (sseSplit+ sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList) #update the cluster assignments
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print('the bestCentToSplit is: ', bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0, :]
        centList.append(bestNewCents[1, :])
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
    return centList, clusterAssment

def geoGrab(strAddress, city):
    apiStem = 'http://where.yahooapis.com/geocode?'
    params = {}
    params['flags'] = 'J'
    params['appid'] = 'ppp68N8t'
    params['location'] = '%s %s' % (strAddress, city)
    url_params = urllib.parse.urlencode(params)
    yahooApi = apiStem + url_params
    print(yahooApi)
    c = urllib.request.urlopen(yahooApi)
    return json.loads(c.read())

    def massPlaceFind(fileName):
        fw = open('places.txt', 'w')
        for line in open(fileName).readlines():
            line = line.strip()
            lineArr = line.split('\t')
            retDict = geoGrab(lineArr[1], lineArr[2])
            if retDict['ResultSet']['Error'] == 0:
                lat = float(retDict['ResultSet']['Results'][0]['latitude'])
                lng = float(retDict['ResultSet']['Results'][0]['longitude'])
                print('%s\t%f\t%f' % (lineArr[0], lat, lng))
                fw.write('%s\t%f\t%f\n' % (line, lat, lng))
            else:
                print('error fetching')
                sleep(1)
        fw.close()


if __name__ == '__main__':
    # datMat = mat(loadDataSet('testSet2 _kmeans.txt'))
    # centList, myNewAssments = biKmeans(datMat,3)
    # print(centList)
    # myCentroids, clustAssing = kMeans(datMat, 4)
    # print(randCent(datMat, 2))
    geoResult = geoGrab('1 VA Center', 'Augusta, ME')
    print(geoResult)
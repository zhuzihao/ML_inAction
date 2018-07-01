from math import log
import operator

# 熵为信息的期望值
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)           # 数据集条目
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries       # 计算标签出现概率
        shannonEnt -= prob*log(prob, 2)           # x信息定义为 -log_2(p(x))
    return shannonEnt


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['浮出水面不能生存', '有脚蹼']
    return dataSet, labels


def splitDataSet(dataSet, axis, value):
    retDateSet = []
    for featVec in dataSet:
        if featVec[axis] == value:          # 找到特征axis等于value的
            reducedFeatVec = featVec[:axis]             # 这边是剔除这个特征
            reducedFeatVec.extend(featVec[axis+1:])     # extend是逐个逐个加
            retDateSet.append(reducedFeatVec)           # append为整体作为单个元素加入
    return retDateSet


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1           # dataSet[0]为一条数据，-1因为包含label
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1;
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]      # 为数据集中特征i的所有特征值
        uniqueVals = set(featList)          # 除去重复的特征值
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)            # 以特征i为value划分数据集
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)         # 特征i给定的情况下的经验条件熵
        infoGain = baseEntropy - newEntropy         # 信息增益
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return  bestFeature


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys() : classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),           # 以字典第二项排序，即次数
                              key=operator.itemgetter(1), reverse=True)         # 降序
    return  sortedClassCount[0][0]


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]        # 数据集的标签
    if classList.count(classList[0]) == len(classList):         # 只有一类数据
        return  classList[0]
    if len(dataSet[0]) == 1:        # 没有特征用于划分，选取次数最多的
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree
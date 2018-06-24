from numpy import *
import operator
from os import listdir          # 指定目录文件


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]          # numpy中shape表示各维度，这里shape[0]代表了数据集在维度0上的大小
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet         # 这里tile 将代分类的 复制个数与训练集一样，并计算之间差值（类似求x1-x2,y1-y2）
    sqDiffMat = diffMat ** 2           # 简单平方差值 类似求(x1-x2)^2  (y1-y2)^2
    sqDistances = sqDiffMat.sum(axis=1)         # axis 从0开始，指定相加的维度， 这里类似 (x1-x2)^2+(y1-y2)^2
    distances = sqDistances ** 0.5          # 这里开根号，求得欧式距离
    sortedDistIndicies = distances.argsort()            # argsort返回的是排序完的索引array
    classCount = {}         # 空字典用来存对应类别的次数
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]          # 按照排序好的顺序获取前k个的类别标签
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1          # get(a, [default])，没有a返回default
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1), reverse=True)         # item()分解为可迭代tuple，itemgetter获取对象第几维，降序
    return sortedClassCount[0][0]


def file2matrix(filename):
    fr = open(filename)        # 读取文件没有异议
    arrayOLines = fr.readlines()            # readlines返回文件所有行保存在list中
    numberOLines = len(arrayOLines)         # list的长度
    returnMat = zeros((numberOLines, 3))            # 返回numberOLines行3列的值为0的array
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()         # strip不含参数 移除字符串首尾空白
        listFromLine = line.split('\t')         # 以\t划分字符串为list
        returnMat[index, :] = listFromLine[0:3]         # ，为ndarray的切片，前面表示第index行，后面表示所有列(一行行的赋值)
        classLabelVector.append(int(listFromLine[-1]))          # -1为倒数第一个，即对应label
        index += 1          # 对应的是matrix的行标
    return returnMat, classLabelVector


def autoNorm(dataSet):
    minVals = dataSet.min(0)            # 0为选每列最小，维数为列
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))         # 创建和dataSet一样的0数组
    m = normDataSet.shape[0]            # shape为tuple, m为行
    normDataSet = dataSet - tile(minVals, (m, 1))           # 计算所有和最小的差值
    normDataSet = normDataSet/tile(ranges, (m, 1))           # 归一化
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minvals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)            # 测试集占10%
    errorCount = 0.0
    for i in range (numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],          # 切片前10%为测试，后90%为训练
                                     datingLabels[numTestVecs:m], 3)                    # 选取k为3
        print("分类结果为: %d, 真实结果为: %d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("总错误率为: %f" % (errorCount/float(numTestVecs)))


def classifyPerson():
    resultList = ['不喜欢', '一般', '喜欢']
    percentTats = float(input("玩电视游戏的时间？"))
    ffMiles = float(input("飞行里程数？"))
    iceCream = float(input("吃冰激凌数?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat ,ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0(inArr-minVals/ranges, normMat, datingLabels, 3)
    print("你喜欢的可能性？ ", resultList[classifierResult])


def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()         # 读取行
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect


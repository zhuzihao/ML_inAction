from math import log

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
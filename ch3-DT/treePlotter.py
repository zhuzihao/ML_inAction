import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="sawtooth",  fc="0.8")         # 定义3个dict用于存特征
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',                     # 用于标注，xy坐标点，
                            xytext=centerPt, textcoords='axes fraction',                        # 注解位置
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)     # 水平 垂直对齐位置

def createPlot():
    fig = plt.figure(1, facecolor='white')          # 创建一个背景颜色白色的figure
    fig.clf()           # 清除figure
    createPlot.ax1 = plt.subplot(111, frameon=False)            # 用于将画布分成1*1个区域，在1区作图， false禁止画图框
    plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()

def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = myTree.keys()[0]         # 获取根节点
    secondDict = myTree[firstStr]           # 获取根节点的子树
    for key in secondDict.keys():           # 判断子树是树还是叶子
        if type(secondDict[key]).__name__=='dict':          # 是树递归调用
            numLeafs += getNumLeafs(secondDict[key])
        else:   numLeafs += 1           # 是叶子+1
    return numLeafs

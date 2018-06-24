import kNN
import matplotlib.pyplot as plt
import matplotlib

"""   
group, labels = kNN.createDataSet()

print(kNN.classify0([0,0], group, labels, 3))
"""
# datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')
#
# fig = plt.figure()          # 创建一个新figure
# ax = fig.add_subplot(111)           # 创建子图，参数决定位置
# ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2],
#            15.0*kNN.array(datingLabels), 15.0*kNN.array(datingLabels))            # 将第一个和第二个特征以散列形式画出,大小，颜色
# plt.show()                  # 创建figure之后才显示
#
#
# normMat, ranges, minVals = kNN.autoNorm(datingDataMat)
# print(normMat)
# print(ranges)
# print(minVals)

# kNN.classifyPerson()
#
# testVector = kNN.img2vector('testDigits/0_1.txt')
# print(testVector[0, 0:31])

kNN.handwritingClassTest()
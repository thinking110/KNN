

#coding=utf-8
import  numpy  as np 

from numpy import *

import  os ,sys 





"""
将一个图片从32*32 变成   1*1024
"""
def   TextToList(fileName):
     
      tmpFile = open(fileName)

      line =tmpFile.readline()
      tmpCount=0

      dataSet = []
      while line:
        tmpCount= tmpCount+1
        for i  in range(32):
            dataSet.append(int(line[i]))
        line = tmpFile.readline()
        pass
      tmpFile.close()
      #print("tmpCount==",tmpCount)
      #print(dataSet)
      return  dataSet


"""
读取所有的图片文件  变成 1935*1024 矩阵
"""
def  ReadFiles(myPath):
    fileList= os.listdir(myPath)

    #print("fileList==",fileList)

    m = len (fileList)
    tranDataSet = np.zeros((m,1024))

    trainLabels = np.zeros((1,m))

    for  i  in  range(m) :
         
         tmpName = fileList[i][0]
         
         trainLabels[0,i]= tmpName
         #print(trainLabels[1,i])

         oneFile = TextToList(myPath +"/"+fileList[i])

         tranDataSet[i,] = oneFile

         #print(tranDataSet[i,],len(tranDataSet[i,]))
         #print(oneFile,len(oneFile))
    
    return  tranDataSet,trainLabels


"""
读取历史图片  变成 1935*1024 矩阵
"""
def  CreateData():
    
    myPath= "./trainingDigits"
    return  ReadFiles(myPath)

"""
读取 将要预测的图片
"""
def  ReadTest():

    myPath= "./testDigits"
    return  ReadFiles(myPath)



"""
将预测矩阵 变成  1935*1024   和历史数据 矩阵1934*1024
做欧式距离判断  并根据前K 个值判断  所属分类
"""
def   Classify(inX,dataSet,labels,k):

      tmpShape= dataSet.shape

      print("shape ==",dataSet.shape,inX.shape)
      differMat = np.tile(inX,(tmpShape[0],1))

      print("differMat==",differMat.shape)

      differMat = mat(differMat) - dataSet

      #print("differMat22==",differMat)

      differMat =  square(differMat) 
      #multiply(differMat,differMat)

      #print("differMat **2==",differMat)

      sqDistance =  differMat.sum(axis=1)

      #print("sqDistance ==",sqDistance)

      sqDistance = sqrt(sqDistance)
      #print("sqDistance =22=",sqDistance)

      sortDistance = np.array(sqDistance.argsort(axis=0)) 
      print("sortDistance ==",sortDistance)

      #print(sortDistance[2][0])
     
      label= np.array(labels.T)

      classCount= {}
      for i  in range(k):
          tmpIndex = sortDistance[i][0]
          voteLable = label[tmpIndex][0]
          #print(tmpIndex, label[tmpIndex],voteLable)
          if(not classCount.get(voteLable)):
               classCount[voteLable] =0
          classCount[voteLable] +=1

      #print ("classCount ==",classCount)
      sortedClass = sorted(classCount.items())

      print ("sorted ==",sortedClass[0][0])

      return  sortedClass[0][0]







"""
所有的流程 和 计算错误率
"""
def  Test():

    trainData,trainLabels = CreateData()

    testData,testLables= ReadTest()

    realRate= {}

    errorRate = {}

    for x in  range(len(testData)):
        testOne =Classify(testData[x,],trainData,trainLabels,1)

        if  not realRate.get(testLables[0,x]):
            realRate[testLables[0,x]] = 0
            errorRate[testLables[0,x]] =0

        if( testOne != testLables[0,x]):
            errorRate[testLables[0,x]] +=1.0
        realRate[testLables[0,x]] +=1.0 


    for key  in realRate.keys():

        tmpFloat = errorRate[key] / realRate[key]

        print("number==",key,"errorRate ==",tmpFloat)



Test()




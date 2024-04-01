import time
import sys
import os
import cv2
import numpy as np
from BroadLearningSystem import BLStrain, BLS_AddEnhanceNodes, BLS_AddFeatureEnhanceNodes, bls_train_input, bls_train_inputenhance
from sklearn.decomposition import PCA

def MidBlock_PCA(matrix):
    # pca = PCA(n_components=0.90)
    pca = PCA(n_components='mle')
    mid = pca.fit_transform(matrix)
    mid = pca.inverse_transform(mid)
    # mid = mid.astype(np.uint8)
    return mid

#此时类别是5个类
def transformLabel(label_raw):
    if label_raw == 0: ##
        results = [1, 0, 0, 0, 0]
    elif label_raw == 1:
        results = [0, 1, 0, 0, 0]
    elif label_raw == 2:
        results = [0, 0, 1, 0, 0]
    elif label_raw == 3:
        results = [0, 0, 0, 1, 0]
    else:
        results = [0,0, 0, 0, 1]
    return results
# CelebA
def transformLabel_CelebA(label_raw):
    if label_raw == 0: ##
        results = [1, 0]
    else:
        results = [0, 1]
    return results


def getFeature(filePath, resize_format = (224, 224), resize_interpolation=cv2.INTER_LANCZOS4):
    tmpData = []
    tmpLabel = []
    with open(filePath) as txtData:
        lines = txtData.readlines()
        for line in lines:
            file, label = line.strip().split() 
            label=int(label)####将string转换为int
            tmpLabel.append(transformLabel(label))
            # tmpLabel.append(transformLabel_CelebA(label))
            fileName = file
            img = cv2.imread(fileName, 1)
            img_formated = cv2.resize(img, resize_format, interpolation=resize_interpolation)
            img_formated = np.expand_dims(img_formated, axis=0)
            img_flat = img_formated.ravel()
            tmpData.append(img_flat)
    return np.double(tmpData), np.double(tmpLabel)####返回样本和标签


def make_print_to_file(path='./'):
    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8',)
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        def flush(self):
            pass
    fileName = "BLS_SCUT-FBP5500"
    # fileName = "PCA+BLS_LSAFBD"
    sys.stdout = Logger(fileName + '.log', path=path)
    print(fileName.center(60,'*'))

make_print_to_file(path='./results')

if __name__ == '__main__':
    
    t1=time.time()
    
    # 获取数据
    #filePath_train = r'/media/xie/F/XieXS/datasets/LSAFBD/train.txt'
    #filePath_test = r'/media/xie/F/XieXS/datasets/LSAFBD/test.txt'

    # filePath_train = r'/media/xie/F/XieXS/datasets/CelebA/train.txt'
    # filePath_test = r'/media/xie/F/XieXS/datasets/CelebA/test.txt'

    # filePath_train = r'/media/xie/F/XieXS/datasets/SCUT-FBP/train.txt'
    # filePath_test = r'/media/xie/F/XieXS/datasets/SCUT-FBP/test.txt'

    filePath_train = r'/media/xie/F/XieXS/datasets/SCUT-FBP5500/train.txt'
    filePath_test = r'/media/xie/F/XieXS/datasets/SCUT-FBP5500/test.txt'

    traindata, trainlabel = getFeature(filePath_train)
    testdata, testlabel = getFeature(filePath_test)
    

    np.save(r"./data/SCUT-FBP5500/traindata.npy", traindata) # save feautres
    np.save(r"./data/SCUT-FBP5500/trainlabel.npy", trainlabel) # save labels
    np.save(r"./data/SCUT-FBP5500/testdata.npy", testdata)
    np.save(r"./data/SCUT-FBP5500/testlabel.npy", testlabel)   

    traindata = np.load("./data/SCUT-FBP5500/traindata.npy",encoding = "bytes") # load feautres
    trainlabel = np.load("./data/SCUT-FBP5500/trainlabel.npy",encoding = "bytes") # load labels
    testdata = np.load("./data/SCUT-FBP5500/testdata.npy",encoding = "bytes")
    testlabel = np.load("./data/SCUT-FBP5500/testlabel.npy",encoding = "bytes")

    # traindata = MidBlock_PCA(traindata) # PCA
    # testdata = MidBlock_PCA(testdata) # PCA

    # # 参数设置
    N1 = 14
    N2 = 50
    N3 = 3500
    s = 0.10
    C = 2**-10

    # incremental learning
    # adding new nodes
    # N1 = 8
    # N1 = 10
    # N1 = 12
    # N1 = 14
    # N1 = 16

    # N2 = 50

    # N3 = 1500
    # N3 = 2000
    # N3 = 2500
    # N3 = 3000
    # N3 = 3500


    # s = 0.10
    # C = 2**-10
    # L = 5
    # M1 = 20
    # M2 = 50
    # M3 = 250

    # adding new data
    
    # N1 = 14
    # N2 = 50
    # N3 = 3450
    # s = 0.10
    # C = 2**-10
    # l = 5
    # m = 2000

    t1=time.time()
    for i in range(1):
        print("*****************start*******************")
        print('-------------------BLS---------------------------')
        BLStrain(traindata, trainlabel, testdata, testlabel, s, C, N1, N2, N3)
        # print('-------------------BLS_AddEnhanceNodes------------------------')
        # BLS_AddEnhanceNodes(traindata, trainlabel, testdata, testlabel, s, C, N1, N2, N3, L, M1)
        # print('-------------------BLS_AddFeatureEnhanceNodes----------------')
        # BLS_AddFeatureEnhanceNodes(traindata, trainlabel, testdata, testlabel, s, C, N1, N2, N3, L, M1, M2, M3)
        # print('-------------------BLS_INPUT--------------------------')
        # bls_train_input(traindata[0:84533,:],trainlabel[0:84533,:],traindata,trainlabel,testdata,testlabel,s,C,N1,N2,N3,l,m)
        # print('-------------------bls_train_inputenhance--------------------------')
        # bls_train_inputenhance(traindata[0:10000,:],trainlabel[0:10000,:],traindata,trainlabel,testdata,testlabel,s,C,N1,N2,N3,l,m,m2)
        print("*****************end*******************")   
    t2=time.time()
    traintime=t2-t1
    print("traintime is:",traintime)
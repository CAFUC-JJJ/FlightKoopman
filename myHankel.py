"""""
by lujing
lujing_cafuc@nuaa.edu.cn

"""""

import numpy as np
import scipy.linalg
import tensorflow as tf

#######################################################################
# Hankel time delay embeddding
# var = scipy.linalg.hanekl(c,r=None)
# 参数：
# c： array_like    矩阵的第一列。无论c的实际形状如何，它将被转换为一维数组。
# r： 数组，可选    矩阵的最后一行。如果没有，则假定为r = zeros_like(c)。 r[0]被忽略；返回矩阵的最后一行是[c[-1], r[1:]] 。无论r的实际形状如何，它将被转换为一维数组。
# 返回：
# A： (len(c), len(r))ndarray
def myHankel(inputdata,epsilon,d,name):
    inputdataLen = len(inputdata)
    ee = (epsilon + 1) * (epsilon + 1)
    # hankelNo_train = math.floor(inputdataLen*d / ee)
    hankelNo = inputdataLen - d + 2  # 2023-08-30
    print("------------------------------------------------")
    print("Hankel for ",name," is begin:")
    print(name,"_hankelNo:", hankelNo)
    XX = inputdata
    hankelNo = hankelNo
    Gamma = np.array([])
    # i = 0
    # j = 0
    # mm = 0
    # nn = 0
    for i in range(hankelNo):
        # c=np.array(XX[i])
        # r=np.array(XX[i+epsilon])
        # print('第%d个块:' % i)
        for j in range(epsilon + 1):
            if j == 0:
                c = np.array(XX[i + j]).reshape(1, -1)
                r = np.array(XX[i + epsilon + j]).reshape(1, -1)
            else:
                # c = np.append(c,XX[i+j])
                c = np.concatenate((c, XX[i + j].reshape(1, -1)), axis=0)
                r = np.concatenate((r, XX[i + epsilon + j].reshape(1, -1)), axis=1)
            # print('XX[i+j]:', XX[i + j])
            # print('XX[i+epsilon+j]', XX[i + epsilon + j])
            # print("c:X%d r:X%d" % (i + j, i + epsilon + j))
            # print("i=", i, "j=", j, "c:", c.shape, "r:", r.shape,)
            # print('c:',c,'r:',r)
        G_t = np.matrix(scipy.linalg.hankel(c, r))
        if i==hankelNo-1:
            print('最后一块——第%d个块:' % i,"c:X%d r:X%d" % (i + j, i + epsilon + j))
        # print("G_t:", G_t)
        # print('G_t.shape:', G_t.shape)
        mm, nn = G_t.shape
        if i == 0:
            Gamma = G_t
        else:
            Gamma = np.concatenate((Gamma, G_t), axis=0)
        # print(Gamma_t)
        # print('Gamma_train:\n', Gamma_train)
    print(name,'shape:', Gamma.shape)
    namefile="output/out_"+name+".txt"
    ##输出到txt文件
    output_path = namefile
    with open(output_path, 'w', encoding='utf-8') as file1:
        print(Gamma, file=file1)

    print("Hankel  is done.")

    return Gamma

######################################################################### 经过测试可知，每个块取第一列和最后一行除了第一个元素即可  如 第一列：x0 x1 x2  最后一行x2 x3 x4 取后面的x3 x4即可
##hankel 逆变换思路，取每一个块的第一个元素0行0列，取最后一个块的第一列和最后一行除了第一个元素外的  flag=0为普通变换，flag=1为预测后的变换 直接取第一列
def myHankelRerverse(gammaData,initialData,epsilon,d):
    ##由于hankel变换将字典元素铺平了 需要将每个元素的d维进行重建
    # 逆变换思路，取每一个块的第一个元素0行0列，取最后一个块的第一列和最后一行除了第一个元素外的
    # 具体来说 对于d=6的 取每一个块的前6行0列，取最后一个块的第一列和最后一行除了前d以外的
    # d=6 epsilon = 2  # hankel矩阵维度：(epsilon+1)*d * (epsilon+1)*d
    # 第0个块:
    # c:X0 r:X2
    # c:X1 r:X3
    # c:X2 r:X4
    # G_t.shape: (18, 18)
    # 第1个块:
    # c:X1 r:X3
    # c:X2 r:X4
    # c:X3 r:X5
    # G_t.shape: (18, 18)
    # .......
    # 第5个块：
    # c:X5 r:X7
    # c:X6 r:X8
    # c:X7 r:X9
    # G_t.shape: (18, 18)
    # Gamma_train.shape: (108, 18)
    ####输入用Gamma_train 因为模型计算后大小格式应与此一致
    inputDataForHankelReverse = gammaData  # hankel变换后的
    initilDataForHankelReverse = initialData  # 原始数据的
    print("------------------------------------------------")
    print("Hankel Reverse is begin:")
    print('inputDataForHankelReverse.shape: ',inputDataForHankelReverse.shape)
    print('initilDataForHankelReverse.shape:', initilDataForHankelReverse.shape)
    ##第一步 分块还原 将Gamma_train还原为块
    ###第二步 取每一个块的的前d行0列，同时还原为d维的元素
    ##取最后一个块的第一列和最后一行从位置d开始的，同时还原为d维的元素
    ##flag=0为普通变换，flag=1为预测后的变换 直接取第一列

    batNo = int(inputDataForHankelReverse.shape[0] / ((epsilon + 1) * d))
    print("batNo:", batNo)
    i = 0
    X_res = np.array((initilDataForHankelReverse.shape[0], d))
    for i in range(batNo):
        # start = i * batNo
        # end = (i + 1) * batNo - 1
        start = i * ((epsilon + 1) * d)  # 分割大块
        end = (i + 1) * ((epsilon + 1) * d) - 1
        # print('i%d:start%d, end%d' % (i, start, end))
        res_bat = inputDataForHankelReverse[start:end + 1, :]
        #
        if i == 0:
            X_res = res_bat[0:d, 0].reshape(-1, d)
            xxtemp = X_res
            # print('i:', i, '///xxtemp', xxtemp)
            # print(X_res)
        else:
            if i == (batNo - 1):
                # 取0列分成3个元素
                xxtemp = res_bat[:, 0].reshape(-1, d)
                # print('i:', i, '///xxtemp', xxtemp)
                X_res = np.concatenate((X_res, res_bat[:, 0].reshape(-1, d)), axis=0)
                # 取最后一行从位置d开始的
                xxtemp = res_bat[-1, d:].reshape(-1, d)
                # print('i:', i, '///xxtemp', xxtemp)
                X_res = np.concatenate((X_res, res_bat[-1, d:].reshape(-1, d)), axis=0)
            else:
                # 取每一个块的的前d行0列
                xxtemp = res_bat[0:d, 0].reshape(-1, d)
                # print('i:', i, '///xxtemp', xxtemp)
                X_res = np.concatenate((X_res, res_bat[0:d, 0].reshape(-1, d)), axis=0)
        # print('i:',i,'///X_res',X_res)
        # print(res_bat)
    print('X_res.shape', X_res.shape)
    ##输出到txt文件
    output_path = "output/out_X_res.txt"
    with open(output_path, 'w', encoding='utf-8') as file2:
        print(X_res, file=file2)

    print("Hankel Reverse is done.")

    return X_res



def myHankelTensor(inputdata,epsilon,d,name):
    X = inputdata
    inputdataLen = len(X)
    hankelNO=inputdataLen-d-2
    ee1 = epsilon+1
    ee2 = (epsilon+1)*d
    tensor_size = [hankelNO, ee2, ee2]

    Gamma = tf.Variable(tf.ones(tensor_size, dtype=tf.float32))
    gamma = np.array([])

    for i in range(hankelNO):
        for j in range(ee1):
            if j == 0:
                c = np.array(X[i+j]).reshape(1, -1)
                r = np.array(X[i+epsilon+j]).reshape(1, -1)
            else:
                c = np.concatenate((c, X[i + j].reshape(1, -1)), axis=0)
                r = np.concatenate((r, X[i + epsilon + j].reshape(1, -1)), axis=1)
        gamma = np.matrix(scipy.linalg.hankel(c, r))
        g = tf.constant(gamma, dtype=tf.float32)
        Gamma[i].assign(gamma)
    print(name, ' shape:', Gamma.shape)
    return Gamma

def myHankelRerverse_Tensor(gammaData, initialData, epsilon, d):
    # 由于hankel变换将字典元素铺平了 需要将每个元素的d维进行重建
    # 逆变换思路，取每一个块的第一个元素0行0列，取最后一个块的第一列和最后一行除了第一个元素外的
    # 具体来说 对于d=6的 取每一个块的前6行0列，取最后一个块的第一列和最后一行除了前d以外的
    # d=6 epsilon = 2  # hankel矩阵维度：(epsilon+1)*d * (epsilon+1)*d
    # 第0个块:
    # c:X0 r:X2
    # c:X1 r:X3
    # c:X2 r:X4
    # G_t.shape: (18, 18)
    # 第1个块:
    # c:X1 r:X3
    # c:X2 r:X4
    # c:X3 r:X5
    # G_t.shape: (18, 18)
    # .......
    # 第5个块：
    # c:X5 r:X7
    # c:X6 r:X8
    # c:X7 r:X9
    # G_t.shape: (18, 18)
    # Gamma_train_Tensor.shape: (t,18, 18)
    # 输入用Gamma_train 因为模型计算后大小格式应与此一致
    inputDataForHankelReverse_Tensor = gammaData  # hankel变换后的
    initilDataForHankelReverse = initialData  # 原始数据的
    print("------------------------------------------------")
    print("Hankel Reverse Tensor is begin:")
    print('inputDataForHankelReverse.shape: ',inputDataForHankelReverse_Tensor.shape)
    print('initilDataForHankelReverse.shape:', initilDataForHankelReverse.shape)
    ##第一步 分块还原 将Gamma_train还原为块
    ###第二步 取每一个块的的前d行0列，同时还原为d维的元素
    ##取最后一个块的第一列和最后一行从位置d开始的，同时还原为d维的元素
    ##flag=0为普通变换，flag=1为预测后的变换 直接取第一列

    batNo = int(inputDataForHankelReverse_Tensor.shape[0])
    print("batNo:", batNo)
    i = 0
    X_res = np.array((initilDataForHankelReverse.shape[0], d))
    for i in range(batNo):
        res_bat = inputDataForHankelReverse_Tensor[i,:, :]
        if i == 0:
            X_res = res_bat[0:d, 0].reshape(-1, d)
            xxtemp = X_res
        else:
            if i == (batNo - 1):
                # 取0列分成3个元素
                xxtemp = res_bat[:, 0].reshape(-1, d)
                # print('i:', i, '///xxtemp', xxtemp)
                X_res = np.concatenate((X_res, res_bat[:, 0].reshape(-1, d)), axis=0)
                # 取最后一行从位置d开始的
                xxtemp = res_bat[-1, d:].reshape(-1, d)
                # print('i:', i, '///xxtemp', xxtemp)
                X_res = np.concatenate((X_res, res_bat[-1, d:].reshape(-1, d)), axis=0)
            else:
                # 取每一个块的的前d行0列
                xxtemp = res_bat[0:d, 0].reshape(-1, d)
                # print('i:', i, '///xxtemp', xxtemp)
                X_res = np.concatenate((X_res, res_bat[0:d, 0].reshape(-1, d)), axis=0)
        # print('i:',i,'///X_res',X_res)
        # print(res_bat)
    print('X_res.shape', X_res.shape)
    if X_res.shape[0] < initilDataForHankelReverse.shape[0]:
        delt=initilDataForHankelReverse.shape[0]-X_res.shape[0]
        j=X_res.shape[0]
        for i in range(delt):
            temp = initilDataForHankelReverse[j+i,:].reshape(-1,d)
            X_res = np.concatenate((X_res, temp), axis=0)
        print('X_res.shape(fill)', X_res.shape)

    # ##输出到txt文件
    # output_path = "output/out_X_res.txt"
    # with open(output_path, 'w', encoding='utf-8') as file2:
    #     print(X_res, file=file2)
    print("Hankel Reverse Tensor is done.")
    return X_res



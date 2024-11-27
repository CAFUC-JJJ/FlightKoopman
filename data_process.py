"""""
by lujing
lujing_cafuc@nuaa.edu.cn

"""""
################################
import errno
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import os
import shutil
import pandas as pd
import sklearn
################################
# 张量转矩阵
def ten2mat(tensor, mode):
    mode=mode-1
    x = np.moveaxis(tensor,mode,0)
    s = tensor.shape[mode]
    mm = np.reshape(x, (s, -1), order = 'F')
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order = 'F')

# 矩阵转张量
def mat2ten(mat, tensor_size, mode):
    index = list()
    index.append(tensor_size[mode])
    for i in range(tensor_size.shape[0]):
        if i != mode:
            index.append(tensor_size[i])
    return np.moveaxis(np.reshape(mat, list(index), order = 'F'), 0, mode)

def mode1Unfolding(X):
    mode=1
    M = ten2mat(X, mode)
    return M
def mode3Unfolding_T(X):
    mode=3
    M = ten2mat(X, mode)
    M = M.T
    return M

def mode1Folding(X,tensor_size):
    T1= mat2ten(X, tensor_size, 0)
    return T1
################################
# 遍历目录树
def traversal_dirs(path):
    dirs=[]
    files=[]
    for item in os.scandir(path):
        if item.is_dir():
          dirs.append(item.path)
        elif item.is_file():
          files.append(item.path)
    return dirs

def traversal_files(path):
    dirs=[]
    files=[]
    for item in os.scandir(path):
        if item.is_dir():
          dirs.append(item.path)
        elif item.is_file():
          files.append(item.path)
    return files

def prepareData(inDir, outDir, Infname,colNs,Outfname,fileIndex):
    rawfilename = inDir + '/' + Infname
    oldfile = inDir + '/' +Infname +  ".csv"
    newfile = outDir + Outfname + ".csv"

    if not os.path.exists(os.path.dirname(outDir)):
        try:
            os.makedirs(os.path.dirname(outDir))
        except OSError as exc:  # Guard against race condition of path already existing
            if exc.errno != errno.EEXIST:
                raise
    # 复制原始文件
    rawfilename = inDir + Infname
    olddfile = inDir + '/' + Infname + ".csv"
    newwfile = outDir + "rawData_" +Outfname+ ".csv"
    shutil.copyfile(olddfile, newwfile)

    data = pd.read_csv(oldfile, skiprows=[0, 1], skipinitialspace=True)
    data.fillna(0.0, inplace=True)
    i = 0
    col_names = data.columns.tolist()
    # 把列名中的空格去掉
    for index, value in enumerate(col_names):
        col_names[index] = value.replace(" ", "")
    # 修改列名字
    data.columns = col_names
    newdata = pd.DataFrame(data,
                           columns=colNs,
                           dtype=float
                           )
    newdata['fileIndex']=fileIndex
    newdata.to_csv(newfile, header=None, index=None)
    return newfile

def outputAll(outDir, inDir, fname, result, numbercluster):
    resultfiename = outDir + "Results.txt"
    np.savetxt(resultfiename, result, fmt='%d', delimiter=',')
    rawfile = outDir + "rawData" + ".csv"
    rawdata = pd.read_csv(rawfile, header=None, error_bad_lines=False)
    puredata = pd.read_csv(rawfile, header=None, skiprows=[0, 1, 2])
    rawheader = pd.DataFrame(rawdata, index=[0, 1, 2])
    cluster = result
    nc = numbercluster
    n = len(cluster)
    i = 0
    j = 1
    # 准备文件头
    tempdata = rawheader
    for i in range(n - 1):
        c = cluster[i]
        e=cluster[i+1]
        filename = outDir + str(j) + "_class" + str(int(c)) + ".csv"
        # 逐行添加数据
        jj=c
        for jj in range(c,e):
            tempdata = tempdata.append(puredata.iloc[jj])
        tempdata.to_csv(filename, index=False, header=False)
        tempdata = rawheader
        j = j + 1  # 文件名累加

    return j

def gaussian(x,mu,sigma):
    f_x = 1/(sigma*np.sqrt(2*np.pi))*np.exp(-np.power(x-mu, 2.)/(2*np.power(sigma,2.)))
    return(f_x)

# 2_dimension gaussian function
def gaussian_2(x,y,mu_x,mu_y,sigma_x,sigma_y):
    f_x_y = 1/(sigma_x*sigma_y*(np.sqrt(2*np.pi))**2)*np.exp(-np.power\
              (x-mu_x, 2.)/(2*np.power(sigma_x,2.))-np.power(y-mu_y, 2.)/\
              (2*np.power(sigma_y,2.)))
    return(f_x_y)

def cordinatesAltProcess(filename):
    data=pd.read_csv(filename)
    X,Y,Z=data.iloc[:,0],data.iloc[:,1],data.iloc[:,2]
    X_center,Y_center,Z_center=X[0],Y[0],Z[0]
    for i in range(len(X)):
        X[i]=(X[i]-X_center)*1000
        Y[i]=(Y[i]-Y_center)*1000
        Z[i]=Z[i]-Z_center
    data.to_csv(filename,header=None,index=None,)
    return filename
def err_order(d, data_res,data_true,flag='<'):
    #######对误差排序
    for i in range(d):
        if i == 0:
            err = np.array(
                (sklearn.metrics.mean_absolute_percentage_error(data_res[:, i], data_true[:, i])))
        else:
            err = np.append(err,
                                  (np.array((sklearn.metrics.mean_absolute_percentage_error(data_res[:, i],
                                                                                            data_true[:, i])))))
    err_d = np.argsort(err)
    if flag=='>':
        err_d = err_d[::-1]
    return  err,err_d

def err_order_Tensor(d, data_res,data_true,flag='<'):
    #######对误差排序
    x= mode3Unfolding_T(data_res)
    y= mode3Unfolding_T(data_true)
    dd=data_res.shape[-1]

    for i in range(dd):
        if i == 0:
            err = np.array(
                (sklearn.metrics.mean_absolute_percentage_error(y[:, i], x[:, i])))
        else:
            err = np.append(err,
                                  (np.array((sklearn.metrics.mean_absolute_percentage_error(y[:, i],
                                                                                            x[:, i])))))
    err_d = np.argsort(err)  ##按照误差值从小到大排序
    if flag=='>':
        err_d = err_d[::-1]  # 逆序
    return  err,err_d


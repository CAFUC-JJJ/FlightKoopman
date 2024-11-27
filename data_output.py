"""""
by lujing
lujing_cafuc@nuaa.edu.cn

"""""
import numpy as np
import pandas as pd


def outputAll(outDir, inDir, fname, result, numbercluster):
    # 输出聚类结果到txt
    # with open
    resultfiename = outDir + "Results.txt"
    np.savetxt(resultfiename, result, fmt='%d', delimiter=',')

    # 在保留头部的情况下，按照聚类分割原始文件并单独存放，例如第1个文件_聚类类别.csv,文件名为1_6.csv等等
    # 头部为前三行，用原始文件切割
    # error_bad_lines=False 忽略其中出现错乱(例如，由于逗号导致多出一列)的行
    rawfile = outDir + "rawData" + ".csv"
    rawdata = pd.read_csv(rawfile, header=None, error_bad_lines=False)
    # rawdata.fillna(0.0,inplace=True)

    puredata = pd.read_csv(rawfile, header=None, skiprows=[0, 1, 2])
    # puredata.fillna(0.0,inplace=True)
    # print(puredata.head())

    rawheader = pd.DataFrame(rawdata, index=[0, 1, 2])
    # rawheader.drop(index=0)
    # print(rawheader)

    cluster = result
    nc = numbercluster
    n = len(cluster)
    # print(n)

    i = 0
    j = 1

    # 准备文件头
    tempdata = rawheader
    for i in range(n - 1):
        c = cluster[i]
        filename = outDir + str(j) + "_class" + str(int(c)) + ".csv"
        # print("i:",str(i),"c:",str(int(c)),"j:",str(j))
        # print(puredata.loc[i])

        # 逐行添加数据
        tempdata = tempdata.append(puredata.loc[i])
        # 下一个数据不是一类或者已到文件结尾，截断输出，重置filename和tempdata
        if (cluster[i + 1] != c):
            # print(tempdata.head())
            tempdata.to_csv(filename, index=False, header=False)
            j = j + 1  # 文件名累加
            tempdata = rawheader

    tempdata.to_csv(filename, index=False, header=False)

    return j


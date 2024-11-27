"""""
by lujing
lujing_cafuc@nuaa.edu.cn

 """""
################################################################
from data_process import *
from data_plot import *

from chaos import *


warnings.filterwarnings("ignore")
np.set_printoptions(threshold=np.inf)
################################################################

# CAFUC datasets
def datasets_CAFUC(datasetsName, dirName, outDir):
    colnamesTotal = ["Latitude", "Longitude", "AltMSL", "OAT", "IAS",
                     "GndSpd", "VSpd", "Pitch", "Roll", "LatAc",
                     "NormAc", "HDG", "TRK", "FQtyL", "E1FFlow",
                     "E1OilT", "E1OilP", "E1RPM", "E1CHT1", "E1EGT1",
                     "VSpdG"]
    colnamesNeed = ["Latitude", "Longitude", "AltMSL", "Pitch", "Roll",
                    "LatAc", "NormAc", "HDG", "TRK", "E1RPM","E1CHT1","E1EGT1"]

    outDirs = outDir + datasetsName + "/"
    dirs = traversal_dirs(dirName + datasetsName)
    dirNo = len(dirs)
    filesNoTotal = 0
    k = 1
    BigData = pd.DataFrame()
    BigDataFilename = outDirs + "bigdata_test.csv"
    return BigData, BigDataFilename , colnamesNeed

def datasets_CAFUC_prepare(dirNo, dirName, datsetsName, outDir, colnamesNeed, BigDataFilename):
    filesNoTotal = 0  # 初始化文件总数
    k = 1
    #data prepare 只需要运行一次
    for i in range(1,dirNo):
        inDir = dirName + datsetsName + "/file" + str(i)
        files=traversal_files(inDir)
        fileNo=len(files)
        filesNoTotal=filesNoTotal+fileNo
        for j in range(fileNo):
            filename=str(j+1)
            outname=str(k)
            fileIndex=inDir + '/' + filename
            f=prepareData(inDir,outDir,filename,colnamesNeed,outname,fileIndex)
            #解决经纬度偏心问题和场高问题 获得绝对经纬度和场高
            ff=cordinatesAltProcess(f)
            ##拼接为一个大数据文件
            BigData=pd.read_csv(ff)
            BigData.to_csv(BigDataFilename,header=None,index=None,mode='a+')
            k=k+1

    return BigData

def dataset_test(n,d):
    tempp = np.arange(0, n)
    inputData = pd.DataFrame(tempp.reshape(-1, d))
    print(inputData.shape)
    return inputData

def dataset_lorenz_attractor(N):
    a, b, c = 10.0, 28.0, 8.0 / 3.0  # a为普兰特数,b为规范化的瑞利数,c与几何形状相关
    h = 0.01  # 微分迭代步长
    x0, y0, z0 = 0.1, 0, 0
    for i in range(N):
        x1 = x0 + h * a * (y0 - x0)
        y1 = y0 + h * (x0 * (b - z0) - y0)
        z1 = z0 + h * (x0 * y0 - c * z0)
        x0, y0, z0 = x1, y1, z1
        xs.append(x0)
        ys.append(y0)
        zs.append(z0)
    lorenz = np.concatenate((np.array(xs).reshape(-1,1),np.array(ys).reshape(-1,1),np.array(zs).reshape(-1,1)),axis=1)
    print("lorenz.shape:",lorenz.shape)
    return lorenz

def dataset_rossler(N):
    a,b,c=0.1,0.1,14
    h=0.01
    x0, y0, z0 = 1, 1, 1
    for i in range(N):
        x1=x0+h*(-y0-z0)
        y1=y0+h*(x0+a*y0)
        z1=z0+h*(b+z0*(x0-c))
        x0, y0, z0 = x1, y1, z1
        xs.append(x0)
        ys.append(y0)
        zs.append(z0)
    rossler = np.concatenate((np.array(xs).reshape(-1,1),np.array(ys).reshape(-1,1),np.array(zs).reshape(-1,1)),axis=1)
    print("rossler.shape:",rossler.shape)
    return rossler



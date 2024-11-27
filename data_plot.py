"""""
by lujing
lujing_cafuc@nuaa.edu.cn

"""""
# -*-coding:utf-8-*-
"""
python绘制标准正态分布曲线
"""
# ==============================================================
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import sklearn
import seaborn as sns
import pandas as pd

def plot_learning_curves(model, X_train, y_train, X_test, y_test,titles):
    train_error,test_error = [],[]
    len_plot=min(len(X_test),3000)
    min_train_error,min_test_error =10000.0,10000.0
    for i in range(1,len_plot):
        y_train_pred = model.predict(X_train[:i])
        y_test_pred=model.predict(X_test[:i])
        temp_train_err=sklearn.metrics.mean_squared_error(y_train[:i], y_train_pred)
        train_error.append(temp_train_err)
        if temp_train_err<min_train_error:
            min_train_error=temp_train_err
        temp_test_err=sklearn.metrics.mean_squared_error(y_test[:i],y_test_pred)
        test_error.append(temp_test_err)
        if temp_test_err<min_test_error:
            min_test_error=temp_test_err

    plt.plot(np.sqrt(train_error), "r-+",linewidth=2,label="train")
    plt.plot(np.sqrt(test_error), "b-",linewidth=3,label="test")

    plt.axhline(y=np.sqrt(min_train_error), c='r',ls='--',lw=2)
    plt.axhline(y=np.sqrt(min_test_error), c='b',ls='--',lw=2)

    plt.legend()
    plt.xlabel("X_train length:"+str(len(X_train)))
    plt.title(titles)
    plt.show()


def gd(x, mu=0, sigma=1):
    left = 1 / (np.sqrt(2 * math.pi) * np.sqrt(sigma))
    right = np.exp(-(x - mu) ** 2 / (2 * sigma))
    return left * right

def gdPlot(x,mu=0,sigma=1,colors='black'):
    # 因变量（不同均值或方差）
    y=gd(x,mu,sigma)
    # 绘图
    l,=plt.plot(x,y,color=colors)
    plt.grid(True)
    return l

def segment_gd_plot(data,bp,feats):  #按照分割好的时序段 绘制正态图 bp是断点 feats是选择列作为自变量
    cmap=mpl.cm.get_cmap("Accent",8)
    colors=cmap(np.linspace(0,1,len(bp)))
    str1=[]
    handlePlot=[]
    listPlot=[]
    for i in range(len(bp)-1):
        i1=bp[i]
        i2=bp[i+1]
        tempdata=data[i1:i2,:]
        x=tempdata[:,feats]
        mu=np.mean(tempdata)
        var=np.std(tempdata)
        cov=np.cov(tempdata.T,bias=True)
        strrr=str(i+1)+':'+str(i1)+'-->'+str(i2)
        listPlot.append(strrr)
        handlePlot.append(gdPlot(x, mu, var, colors=colors[i]))
    plt.legend(handles=handlePlot, labels=listPlot, loc='best')
    return

#plot the results of 2D reconstruction
def plotResults2D_Res(par,data_res ,data_true,d,colnamesNeed=[],model_eqn=[],title=None,hline=None,vline=None,):
    #png name
    ax = []
    leng = 1000
    eqns = model_eqn
    fig_Res = plt.figure(figsize=(12,4),dpi=100)
    #True+Res
    for i in range(d):
        ax.append(plt.subplot(d,1,i+1))
        ax[i].plot(data_true[:, i], 'b')
        ax[i].plot(data_res[:, i], 'r-')
        if i == 0:
            ax[i].set_title(title)
        if hline == None:
            pass
        else:
            plt.axhline(y=hline, c='black',ls='--')
        if vline == None:
            pass
        else:
            plt.axvline(x=vline, c='black',ls='--')
        plt.ylabel('x'+str(i))
    fig_violin=plt.figure(figsize=(12,6))
    data1=pd.DataFrame(data_true)
    data2=pd.DataFrame(data_res)
    plt.subplot(211)
    ax1= sns.violinplot(data=data1)
    plt.ylabel('data_true')
    plt.subplot(212)
    ax2= sns.violinplot(data=data2)
    plt.ylabel('data_res')
    return
#plotting 3D
def plot3D(data_true, data_res, flag='Not CAFUC',title=None,data_ll_true=None,data_ll_res=None):
    fig_3d = plt.figure(figsize=(10,10),dpi=100)
    if flag == 'Not CAFUC':
        X = data_true[:, 1]
        Y = data_true[:, 0]
        Z = data_true[:, 2]
        Xr = data_res[:, 1]
        Yr = data_res[:, 0]
        Zr = data_res[:, 2]
    else:
        X = data_ll_true[:, 1]
        Y = data_ll_true[:, 0]
        Z = data_true[:, 0]
        Xr = data_ll_res[:, 1]
        Yr = data_ll_res[:, 0]
        Zr = data_res[:, 0]
    ax=fig_3d.add_subplot(projection='3d')
    ax.plot(X, Y, Z, 'b')
    ax.plot(Xr, Yr, Zr, '--',color = '#FAA460')

    plt.title(title)
    return





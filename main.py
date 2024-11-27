# This is EKNO code main
# wrote by lujing
# lujing_cafuc@nuaa.edu.cn
# 2023-06-01
#######################################################################
import sklearn.metrics

from myHankel import myHankelTensor
from myObserver import *
from myKoopman import *
from myDatasets import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import  StandardScaler

import warnings
warnings.filterwarnings("ignore")

def print_hi(name):
    print(r'{name}')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #GPU mode
    print("GPUs Available: ", tf.config.experimental.list_physical_devices('GPU'))
    tf.config.set_soft_device_placement(True)

    # Initial parameters
    # k1 = m
    k2 = 30
    k3 = 50
    k4 = 100
    k5 = 937
    k6 = 2500
    k7 = 5000
    k8 = 10000
    k9 = 20000

    epsilon = 2  # hankel矩阵维度：(epsilon+1)*d * (epsilon+1)*d
    d = 12  # d是输入值的维度 1-9
    par = np.arange(d)
    step = 3

    TotalDatalen=k7
    kk_train = TotalDatalen
    kk_test = math.floor(kk_train * 1.2)

    flag_train = True
    flag_hankel = False
    flag_datasets = 'Not CAFUC'
    flag = False
    # flag_hankel = True
    # flag_koopman = True  # 是否重新计算koopman模式，true是重新计算，false是直接调用已保存模式
    # flag_koopman= False #是否重新计算koopman模式，true是重新计算，false是直接调用已保存模式

    # Load the datasets
    datasets = 'CAFUC'
    datasetsName = "RectangeDatasets"  #飞行训练科目

    if datasets == 'CAFUC':
        BigData, BigDataFilename, colNeed = datasets_CAFUC(datasetsName=datasetsName, dirName="dataRaw/",
                                                           outDir="output/")
        dir = 'output/' + datasetsName + '/'
        if(flag == True):
            datasets_CAFUC_prepare(2,"./dataRaw/",datasetsName,dir,colNeed,BigDataFilename)
        print("BigDataFilename", BigDataFilename)
        inputData_raw = pd.read_csv(BigDataFilename)  ##true data  (要不考虑经纬度不参加运算，只考虑可视化？)
        d = 12  # d是输入值的维度 1-12
        par = np.arange(d)
        inputData = inputData_raw.iloc[:, :d]
        flag_datasets = 'CAFUC'

    # input and prepare
    (m, n) = inputData.shape
    print(m, n)

    # Initial the data for train and test
    print('................................Datasets prepare is starting................................')
    #生成训练数据
    #stannd scaler
    stand_scaler = StandardScaler()
    stand_scaler = stand_scaler.fit(inputData)
    inputData_stand = stand_scaler.transform(inputData)
    print('stand_scaler.mean_:', stand_scaler.mean_)

    for i in range(d):
        X_value = inputData_stand[:kk_train, par[i]].reshape(-1, 1)
        X_t = X_value
        if i == 0:
            X = X_t
        else:
            X = np.concatenate((X, X_t), axis=1)
    X_train = X
    print("X_train.shape:", X_train.shape)

    #生成测试数据
    for i in range(d):
        Y_value = inputData_stand[kk_train:kk_test, par[i]].reshape(-1, 1)
        Y_t = Y_value
        if i == 0:
            Y = Y_t
        else:
            Y = np.concatenate((Y, Y_t), axis=1)
    X_test = Y
    print("X_test.shape:", X_test.shape)
    X_ll_train = X_train[:, :2]  # save lat log
    X_ll_test = X_test[:, :2]

    print('................................Datasets prepare is end................................')

    if flag_hankel==True:
        # Hankel Emedding 调用myhankel函数
        Gamma_train_Tensor = myHankelTensor(X_train, epsilon, d, "Gamma_train_Tensor")
        #hankel_test
        Gamma_test_Tensor = myHankelTensor(X_test, epsilon, d, "Gamma_test_Tensor")

        # Observer g learning
        print('................................Observer Gamma Train is starting........................')
        dt = 1
        Gamma_train = mode3Unfolding_T(Gamma_train_Tensor)
        Gamma_test = mode3Unfolding_T(Gamma_test_Tensor)
        model_observer_Tensor = myObserver(Gamma_train_Tensor, dt)
        time_start = time.time()  # 开始计时

        dif_Gamma_train_Tensor = model_observer_Tensor.differentiate(Gamma_train_Tensor, t=dt)  ###求导数
        dif_Gamma_test_Tensor = model_observer_Tensor.differentiate(Gamma_test_Tensor, t=dt)

        res_Gamma_train = model_observer_Tensor.predict(Gamma_train_Tensor)
        res_Gamma_train_Tensor = mode1Folding(res_Gamma_train, np.array(Gamma_train_Tensor.shape))
        err_train_Tensor_hat = sklearn.metrics.mean_squared_error(Gamma_train, res_Gamma_train)

        score_train_Tensor = model_observer_Tensor.score(Gamma_train_Tensor, t=dt)
        err_train_Tensor, err_train_d_Tensor = err_order_Tensor(d, res_Gamma_train_Tensor, dif_Gamma_train_Tensor)
        error_Reconstruct_train_Tensor = err_train_Tensor[err_train_d_Tensor[0]]  ##g观测函数重构误差  导数  取最小的重构误差

        res_Gamma_test = model_observer_Tensor.predict(Gamma_test_Tensor)
        res_Gamma_test_Tensor = mode1Folding(res_Gamma_test, np.array(Gamma_test_Tensor.shape))
        err_test_hat = sklearn.metrics.mean_squared_error(Gamma_test, res_Gamma_test)
        score_test = model_observer_Tensor.score(Gamma_test_Tensor, t=dt)  ##采用的r2_score
        err_test, err_test_d = err_order_Tensor(d, res_Gamma_test_Tensor, dif_Gamma_test_Tensor)
        error_Reconstruct_test = err_test[err_test_d[0]]  ##g观测函数重构误差  导数  取最小的重构误差

        print('score_train_Tensor:', score_train_Tensor, "/score_test:", score_test)
        print('err_train_Tensor:', err_train_Tensor_hat, "/err_test:", err_test_hat)
        print('error_Reconstruct_train_Tensor:', error_Reconstruct_train_Tensor, '/error_Reconstruct_test:',
              error_Reconstruct_test)
        print('res_Gamma_train.shape', res_Gamma_train_Tensor.shape, '/res_Gamma_test.shape', res_Gamma_test.shape)
        time_end = time.time()
        print('predicting time cost:', time_end - time_start, 's')
        X_train_dif_hat = myHankelRerverse_Tensor(res_Gamma_train_Tensor, X_train, epsilon, d)
        X_train_dif = myHankelRerverse_Tensor(dif_Gamma_train_Tensor, X_train, epsilon, d)
        err_real = sklearn.metrics.mean_squared_error(X_train_dif, X_train_dif_hat)
        print("Error_real:", err_real)

        plotResults2D_Res(par, X_train_dif_hat, X_train_dif, d, title="Dif,Score:%.2f" % (score_train_Tensor * 100))
        plot3D(X_train_dif, X_train_dif_hat, flag=flag_datasets, title="Dif",data_ll=None)
        # stannd scaler inverse
        Y_train = stand_scaler.inverse_transform(X_train)
        Y_train_hat = stand_scaler.inverse_transform(X_train_dif_hat)
        plotResults2D_Res(par, data_res=Y_train_hat, data_true=Y_train,
                          d=d, model_eqn=model_observer_Tensor.equations(),
                          title="RealValue,Score:%.2f" % (score_train_Tensor * 100))
        plot3D(Y_train, Y_train_hat, flag=flag_datasets, title="RealValue",data_ll=None)
        Gamma_train = X_train_dif_hat

    else:
        #######################################################################
        # Observer g learning
        print('................................Observer Gamma Train is starting........................')

        #test real value
        if datasets=='CAFUC':
            X_train = X_train[:,2:]
            X_test = X_test[:,2:]
            d=d-2

        plot3D(X_train, X_test, flag=flag_datasets, title="Real Value train&test", data_ll_true=X_ll_train,data_ll_res=X_ll_test)
        plotResults2D_Res(par, X_test, X_train, d, title="Real Value train&test")

        time_start = time.time()  # 开始计时
        model = myObserver(X_train, dt=1)
        input_features = model.feature_names
        output_features = model.model.steps[0][1].get_feature_names(input_features)
        coef = model.model.steps[-1][1].coef_
        print("N=", len(input_features), ": input_features is ", input_features)
        print("N=", len(output_features), ": output_features is ",output_features)
        print("L.shape:", coef.shape)

        dif_X_train = model.differentiate(X_train, t=1)
        dif_X_test = model.differentiate(X_test, t=1)
        dif_X_train_hat = model.predict(dif_X_train)
        X_train_hat = model.predict(X_train)
        X_test_hat = model.predict(X_test)
        score_train = model.score(X_train)
        score_test = model.score(X_test)
        mse_train = sklearn.metrics.mean_squared_error(dif_X_train, X_train_hat)
        mse_train_real = sklearn.metrics.mean_squared_error(X_train,X_train_hat)
        mse_test = sklearn.metrics.mean_squared_error(dif_X_test,X_test_hat)
        print('score_train:', score_train,"/score_test:",score_test)
        print('mse_train:', mse_train,"/mse_test:",mse_test)

        # g_function g_train L
        g_train = mytheta(dif_X_train,input_features,output_features,coef)
        g_train_hat = mytheta(X_train_hat,input_features,output_features,coef)
        g_test = mytheta(dif_X_test,input_features,output_features,coef)
        g_test_hat = mytheta(X_test_hat,input_features,output_features,coef)
        print("X_train.shape:",X_train.shape,"=> g_train.shape:", g_train.shape)
        print("input_features:",input_features)
        print("output_features:",output_features)
        plotResults2D_Res(par,g_train_hat,g_train,d=9,title="g_train")
        time_end = time.time()
        print('predicting time cost:', time_end - time_start, 's')

        plot3D(dif_X_train, X_train_hat,flag=flag_datasets, title="Dif", data_ll_true=X_ll_train,data_ll_res=X_ll_train)
        plotResults2D_Res(par, X_train_hat, dif_X_train, d, title="NO Hankel Dif,Score:%.2f,mse:%.2f" % (score_train * 100,mse_train_real))

        Gamma_train = X_train_hat

    print('................................Observer Gamma Train is end................................')

    time_start = time.time()  # 开始计时
    x0=np.array(X_test[0,:])   #test第一个值 向前预测
    print('x0.shape',x0.shape)
    step_simulation=X_test.shape[0]
    print('step_simulation:',step_simulation)
    t_end_test=step_simulation
    tt=np.arange(0,t_end_test)
    pred_X = model.simulate(x0,tt,integrator="odeint")
    true_X = dif_X_test[:step_simulation,:]
    # 如果有空值则把空值换成平均值
    true_X_no_nan = np.nan_to_num(true_X, nan=np.nanmean(true_X))
    pred_X_no_nan = np.nan_to_num(pred_X, nan=np.nanmean(pred_X))

    err_simulaiton_noK=sklearn.metrics.mean_squared_error(true_X, pred_X)
    print('err_simulaiton_noK:%f for step: %d'%(err_simulaiton_noK,step_simulation))

    X_hat = np.append(dif_X_train_hat, pred_X,axis=0)
    X = np.append(dif_X_train,true_X,axis=0)
    X_hat =X_hat[-300:,:]
    X= X[-300:,:]

    # Koopman K learning
    print('................................Koopman K Train is starting........................')

    X = g_train
    Y = g_test
    pred_train,pred_test=myModel(X,Y,window_width=20,columns=output_features)
    plotResults2D_Res(par,pred_train,g_train,d=10,title='koopman_train')
    plot3D(g_train,pred_train,flag=flag_datasets,title='Koopman_train',  data_ll_true=X_ll_train,data_ll_res=X_ll_train)
    #output to csv
    ffname = 'output/' + datasetsName
    np.savetxt(ffname + '_g_train.csv', np.column_stack((X_ll_train[:, 1], X_ll_train[:, 0], g_train[:, 0])),
               fmt='%.5f', delimiter=',')
    np.savetxt(ffname + '_pred_train.csv', np.column_stack((X_ll_train[:, 1], X_ll_train[:, 0], pred_train[:, 0])),
               fmt='%.5f', delimiter=',')

    np.savetxt(ffname + '_g_test.csv', np.column_stack((X_ll_test[:, 1], X_ll_test[:, 0], g_test[:, 0])),
               fmt='%.5f', delimiter=',')
    np.savetxt(ffname + '_pred_test.csv', np.column_stack((X_ll_test[:, 1], X_ll_test[:, 0], pred_test[:, 0])),
               fmt='%.5f', delimiter=',')

    print('................................Koopman K Train is ending........................')
    plt.show()
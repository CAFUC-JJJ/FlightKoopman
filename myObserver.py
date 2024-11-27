"""""
by lujing
lujing_cafuc@nuaa.edu.cn

"""""
#######################################################################
import time
from myDatasets import *
import pysindy as ps

warnings.filterwarnings("ignore")
np.set_printoptions(threshold=np.inf)
#######################################################################
def myObserver(inputdata,dt):
    #定制模型 训练模型
    stlsq_optimizer = ps.STLSQ(threshold=.01, alpha=.5)
    sr3_optimizer = ps.SR3(threshold=0.5, thresholder='l1', max_iter=100)
    ssr_optimizer = ps.SSR(alpha=.05)
    my_fourier_library = ps.FourierLibrary(n_frequencies=10)
    my_poly_library = ps.PolynomialLibrary(include_bias=False)
    my_library = ps.GeneralizedLibrary([my_poly_library, my_fourier_library])  ##字典采用多项式和傅里叶融合
    optimizer = ps.STLSQ(threshold=8,alpha=1e-3, normalize_columns=True)
    estimator = ps.deeptime.SINDyEstimator(
        optimizer=ssr_optimizer,
        feature_library=my_library
    )
    time_start = time.time()
    estimator.fit(inputdata, t=dt, )
    model = estimator.fetch_model()
    time_end = time.time()
    g_Gamma = model.equations(precision=2)
    model.print()
    print('training time cost:', time_end - time_start, 's')
    return model

def mytheta(inputdata, input_features,output_features, L_coef):

    n = inputdata.shape[0]
    k = len(input_features)
    tt = np.array(input_features)

    Nk = len(output_features)
    gamma = np.zeros((n, Nk))
    output_features = [x for x in output_features if x != ' ']

    for j in range(Nk):
        col = str(output_features[j])
        nn = col.split("x")
        if nn[0] == '':
            nn.pop(0)
            for i in range(len(nn)):
                nn[i]=nn[i].strip()
                nn[i]=nn[i].lstrip()
                if nn[i].isdigit():
                    if i == 0:
                        s = int(nn[i])
                        temp_gamma = inputdata[:, s]
                    else:
                        s = int(nn[i])
                        temp_gamma = temp_gamma * inputdata[:, s]
                else:
                    tempcol=nn[i].split("^")
                    s = int(tempcol[0])
                    pow = int(tempcol[1])
                    temp_gamma = np.power(inputdata[:,s],pow)
        else:
            sincos = nn[0].split("(")
            ss=nn[1].split(")")
            sincos[1]=sincos[1].strip()
            s=int(ss[0])
            if sincos[0] == "sin":
                temp_gamma = np.sin(int(sincos[1])*inputdata[:,s])
            else:
                temp_gamma = np.cos(int(sincos[1])*inputdata[:,s])
        gamma[:,j]=temp_gamma
    outputData = gamma
    return outputData

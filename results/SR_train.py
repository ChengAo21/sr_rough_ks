import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from pysr import PySRRegressor,ParametricExpressionSpec,TemplateExpressionSpec
from sklearn.model_selection import train_test_split


####################################################################         读取数据
try:
    path1 = './csvFiles/rough_surface_statistics.csv'
    rough_dat = pd.read_csv(path1, header = None)
    feature_name = rough_dat.iloc[0, :].values
    data_rough = rough_dat.iloc[1:, 1:-1].values.astype(float)
    labels = rough_dat.iloc[1:, -1].values
except FileNotFoundError:
    print(f"File not found: {path1}")


used_col = [0, 1, 4, 5, 6, 8, 9, 11, 12, 13]
# names_ = [r'$S_x$', r'$k_{avg}$', r'$k_{rms}$', r'$Ra$', r'$I_x$',
# r'$P_o$', r'$E_x$', r'$S_k$', r'$K_u$', r'$k_s$']

all_data = data_rough[:, used_col]
Sx,kavg,krms,Ra,Ix,Po,Ex,Sk,Ku,ks = all_data[:, 0].reshape(-1, 1), all_data[:, 1].reshape(-1, 1), all_data[:, 2].reshape(-1, 1), all_data[:, 3].reshape(-1, 1), all_data[:, 4].reshape(-1, 1), all_data[:, 5].reshape(-1, 1), all_data[:, 6].reshape(-1, 1), all_data[:, 7].reshape(-1, 1), all_data[:, 8].reshape(-1, 1), all_data[:, 9].reshape(-1, 1)
kt = data_rough[:,3].reshape(-1, 1)

input_ft = np.concatenate([Po,Ex,Sk,Sx,kavg],axis=1)
output_ft = ks/krms


# Not used all if want to exclude some samples
id_Not_DNSlc = np.where((labels != 'DNSLC'))[0]
idToDelete = []
idToAdd=[]
if len(idToAdd) > 0:
    id_used_ = np.concatenate([np.setdiff1d(id_Not_DNSlc, idToDelete),idToAdd])
else:
    id_used_ = np.setdiff1d(id_Not_DNSlc, idToDelete)
print(id_used_)


inpt, outpt = input_ft[id_used_, :], output_ft[id_used_, :]
print(f"Input shape: {inpt.shape}, Output shape: {outpt.shape}")
####################################################################    Symbolic Regression

template = TemplateExpressionSpec(
    expressions=["f","g","h","ksi","eta"],
    variable_names = ['Po','Ex','Sk','Sx','kavg',],
    # parameters={"p1":1},
    combine="f(Po)*g(Ex)*h(Sk)*ksi(Sx)*eta(kavg)",
)

model = PySRRegressor(
    niterations=1500,
    output_directory = './sr_results/Temp_/',
    expression_spec=template,
    model_selection="best",
    binary_operators=["+", "-", "*", "/",], 
    unary_operators=['tanh','exp','square','cos'],
    nested_constraints={'exp':{'exp':0,'tanh':0,'square':1,'cos':0},
                        'tanh':{'tanh':0,'exp':0,'square':0,'cos':0},
                        'cos':{'tanh':0,'exp':0,'square':1,'cos':0},
                        'square':{'tanh':0,'exp':1,'square':0,'cos':0},

                        },
    precision=64,
    populations=90,
    population_size=80,
    ncycles_per_iteration=1200, 
    maxsize=60, 
    elementwise_loss='L1DistLoss()',  
    verbosity=1,
    parallelism="multithreading",
    progress=True,
    complexity_of_constants=3,
    # weight_simplify = 0.05,
)

model.fit(inpt, outpt)

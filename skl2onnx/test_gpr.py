import numpy as np
import sklearn
import sklearn.datasets
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
import onnxruntime as rt
import sklearn.preprocessing
import skl2onnx
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def get_cluster_indices(cluster_assignments):
    # Create a dictionary to store indices for each cluster
    cluster_indices_dict = {}

    for i, cluster in enumerate(cluster_assignments):
        if cluster in cluster_indices_dict:
            cluster_indices_dict[cluster].append(i)
        else:
            cluster_indices_dict[cluster] = [i]

    return cluster_indices_dict

def cluster_data(data):
    # train_out is a Matrix of shape n_samples times n_targets
    data = np.copy(data)

    # Normalize along columns
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    max_score = 0
    max_score_num_cluster = 1
    max_cluster = 10

    for i in range(2, max_cluster + 1):
        kmeans = KMeans(n_clusters=i, max_iter=200, n_init='auto')
        R = kmeans.fit(data)
        score = silhouette_score(data, R.labels_)
        if score > max_score:
            max_score = score
            max_score_num_cluster = i

    print("Number of clusters:", max_score_num_cluster)
    kmeans = KMeans(n_clusters=max_score_num_cluster, max_iter=200, n_init='auto')
    R = kmeans.fit(data)
    cluster_indices = get_cluster_indices(R.labels_)
    return cluster_indices, R.labels_, max_score_num_cluster


df = pd.read_csv(r"C:\Users\DELL User\Downloads\eq_1403 (2).csv")
print(df.shape)
X, y = df.iloc[:,:16].to_numpy(), df.iloc[:,16:-1].to_numpy()

# scaler = StandardScaler()
# yt = scaler.fit_transform(y)
# label_indices, cluster_labels, num_clusters = cluster(yt, min_samples=5, eps=5.)
label_indices, cluster_labels, num_clusters = cluster_data(y)


for k, v in label_indices.items():
  print(k, len(v))


# everyting upto this loop including data generation could be julia code
# only this loop would need to be called using PyCall.jl
#py
"""
import skl2onnx
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

def run_gaussian_processes(X, y, label_indices, num_clusters):
    lowest_mae = 1e5
    lowest_mae_onnx_path = ""
    for i in range(num_clusters):
        if len(label_indices[i]) < 2000:
            print(f"Skip cluster {i}")
            continue
        
        X_ = X[label_indices[i]][:1500]
        y_ = y[label_indices[i]][:1500]
        # X_ = sklearn.preprocessing.normalize(X_)
        # y_ = sklearn.preprocessing.normalize(y_)
        X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.1)
        
        # X_scaler = MinMaxScaler() # StandardScaler() 
        # y_scaler = MinMaxScaler() # StandardScaler()
        # X_train = X_scaler.fit_transform(X_train)
        # X_test = X_scaler.transform(X_test)
        # y_train = y_scaler.fit_transform(y_train)
        # y_test = y_scaler.transform(y_test)
        gpr = GaussianProcessRegressor(kernel=RBF()) # normalize_y=True
        
        t0 = time.time()
        gpr.fit(X_train, y_train)
        t1 = time.time()

        mae = mean_absolute_error(y_test, gpr.predict(X_test))
        
        if mae < lowest_mae:
            lowest_mae = mae

            initial_type = [("X", skl2onnx.common.data_types.DoubleTensorType([None, None]))]
            onx64 = skl2onnx.convert_sklearn(gpr, initial_types=initial_type, target_opset=12)

            onnx_path = f"gp_model_{i}.onnx"
            with open(onnx_path, "wb") as f:
                f.write(onx64.SerializeToString())
            
            lowest_mae_onnx_path = onnx_path
            
        print("------")
        print(f"Fit on cluster {i} with {X_train.shape[0]} samples...")
        print(f"Total fitting time: {t1-t0}")
        print(f"MAE on Test Set after fitting: {mae}")
        print(f"MSE on Test Set after fitting: {mean_squared_error(y_test, gpr.predict(X_test))}")
        print("------")
    return lowest_mae_onnx_path
"""

# the julia call could look like this: best_onnx_path = py"run_gaussian_processes"(X, y, label_indices, num_clusters)

# and here we would simulate the original fmu
# and a surrogate fmu using the best performing gpr



#TODO: convert pipeline





"""
sess64 = rt.InferenceSession(
        onx64.SerializeToString(), providers=["CPUExecutionProvider"]
    )

    pred_onx64 = sess64.run(None, {"X": X_test})[0]
    pred_skl = gpr.predict(X_test)

    print(np.allclose(pred_onx64.shape, pred_skl.shape))
"""
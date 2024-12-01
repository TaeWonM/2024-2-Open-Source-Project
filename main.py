import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import kagglehub
import numpy as np

# Download latest version
path = kagglehub.dataset_download("sauravmishraa/waterneeded")

print("Path to dataset files:", path)

DataFrame = pd.read_csv(path + "\TARP.csv")
print("Data type : info method")
print(DataFrame.info())
print("Data arithmetic statistics : describe method")
print(DataFrame.describe())
print("\n")
Purified_DataFrame = DataFrame.drop(
    [
        "Air temperature (C)",
        "Wind speed (Km/h)",
        "Air humidity (%)",
        "Wind gust (Km/h)",
        "Pressure (KPa)",
        "ph",
        "rainfall",
        "N",
        "P",
        "K",
    ],
    axis=1,
)

print("Purified Data type : info method")
print(Purified_DataFrame.info())
print("Purified Data arithmetic statistics : describe method")
print(Purified_DataFrame.describe())
print("\n")

Purified_DataFrame["Time_Boundery"] = 0

for i in range(0, 10):
    mask = (Purified_DataFrame.Time >= i * 11) & (
        Purified_DataFrame.Time <= (i + 1) * 11
    )
    Purified_DataFrame.loc[mask, ["Time_Boundery"]] = i

Purified_DataFrame["Status_index"] = 1
mask = Purified_DataFrame.Status == "ON"
Purified_DataFrame.loc[mask, ["Status_index"]] = 1
mask = Purified_DataFrame.Status == "OFF"
Purified_DataFrame.loc[mask, ["Status_index"]] = 0

print(Purified_DataFrame)

Purified_DataFrame.drop(["Time", "Status"], axis=1, inplace=True)

print(Purified_DataFrame)

Grouped_Purified_DataFrame = Purified_DataFrame.groupby(["Time_Boundery"])

sns.set_style("whitegrid")

ls = []

for key, Group in Grouped_Purified_DataFrame:
    mask = Group.Status_index == 1
    temp = Group.loc[mask, :]
    ls.append(sns.jointplot(kind="kde", data=temp, x="Temperature", y="Soil Moisture"))
    ls[key[0]].fig.suptitle(f"Time_Boundery : {key[0]}")
    plt.show()

ls.clear()

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

model_list = []

for key, Group in Grouped_Purified_DataFrame:
    X = Group[["Soil Moisture", "Temperature", " Soil Humidity"]]
    Y = Group[["Status_index"]]
    X = preprocessing.StandardScaler().fit(X).transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, random_state=10
    )
    model_list.append(KNeighborsClassifier(n_neighbors=5))
    model_list[key[0]].fit(X_train, Y_train)
    Y_hat = model_list[key[0]].predict(X_test)
    knn_metrics = metrics.confusion_matrix(Y_test, Y_hat)
    if key[0] == 9:
        knn_metrics = np.array([[int(knn_metrics[0]), 0], [0, 0]])
    print(key[0])
    print(knn_metrics)
    Percision = int(knn_metrics[0][0]) / (
        int(knn_metrics[0][0]) + int(knn_metrics[0][1])
    )
    print(f"Percision : {Percision}")
    Recall = int(knn_metrics[0][0]) / (int(knn_metrics[0][0]) + int(knn_metrics[1][0]))
    print(f"Recall : {Recall}")
    F1score = (Percision * Recall * 2) / (Percision + Recall)
    print(f"F1score : {F1score}")

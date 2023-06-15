import pandas as pd
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
#Lectura del dataset de entrenamiento y de prueba
dataTest = pd.read_csv("P5.csv")
dataTrain = pd.read_csv("P5_Training.csv")
#Division del dataset en X y Y
X_train = dataTrain.iloc[:,:-1].values
y_train = dataTrain.iloc[:,-1].values
X_test = dataTest.iloc[:,:-1].values
y_test = dataTest.iloc[:,-1].values
#Escalado de datos
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#Se establece una lista con las neuronas y sus respectivas capas que obtuvieron los mejores resultados
numeroCapas =[[5,5],[6,6],[10],[14,14],[15],[16,16],[17],[23,23],[24,24],[25,25],[26,26],[27,27],[31,31,31],[32],[33,33]]
iteraciones=[350,450,550,650,750,850,950,1050,1150]
#Se entrena a la RNA con dichas neuronas iterando sobre un rango de epocas
for capa in numeroCapas:
    print(capa)
    for ite in iteraciones:
        clf = MLPClassifier(random_state=1, max_iter=ite, activation="logistic", hidden_layer_sizes=capa,learning_rate="constant",learning_rate_init=0.15,momentum=0.15,solver="sgd",n_iter_no_change=1150)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning,module="sklearn")
            clf.fit(X_train,y_train)
            print(ite,clf.score(X_test,y_test))
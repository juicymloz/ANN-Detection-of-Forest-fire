import pandas as pd
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
#Lectura del dataset de entrenamiento y de prueba
dataTest = pd.read_csv("P3.csv")
dataTrain = pd.read_csv("P3_Training.csv")
#Division del dataset en X y Y
X_train = dataTrain.iloc[:,:-1].values
y_train = dataTrain.iloc[:,-1].values
X_test = dataTest.iloc[:,:-1].values
y_test = dataTest.iloc[:,-1].values
#Escalado de datos
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#Se establece una lista con las neuronas que obtuvieron los mejores resultados
neuronas = [3,4,5,6,7,9,10,11,14,15,16,17,23,24,25,26,27,28,29,31,32,33]
#Se entrena a la RNA con dichas neuronas iterando sobre un rango numero de capas
for i in neuronas:
    print(i)
    numeroCapas = [[i],[i,i],[i,i,i]]
    for ite in numeroCapas:
        clf = MLPClassifier(random_state=1, max_iter=650, activation="logistic", hidden_layer_sizes=ite,learning_rate="constant",learning_rate_init=0.15,momentum=0.15,solver="sgd",n_iter_no_change=1050)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning,module="sklearn")
            clf.fit(X_train,y_train)
            print(len(ite),clf.score(X_test,y_test))
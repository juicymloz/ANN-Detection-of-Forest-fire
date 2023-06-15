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
#Se entrena la RNA con un rango de 3 a 34 neuronas, y con los demas parametros estaticos
for i in range(3,35):
    clf = MLPClassifier(random_state=1, max_iter=650, activation="logistic", hidden_layer_sizes=(i,),learning_rate="constant",learning_rate_init=0.15,momentum=0.15,solver="sgd",n_iter_no_change=1050)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning,module="sklearn")
        clf.fit(X_train,y_train)
        print(i,clf.score(X_test,y_test))
import sklearn.neighbors as knn
import numpy as np

# ##==============================Realiza a classificação por knn============================

f = np.load('features/A01T_features.npy', allow_pickle=True).item()
features = f['lt']

# Aleatoriza o vetor de atributos
rnd = np.arange(0, 144)
np.random.shuffle(rnd)
features = features[rnd, :]

# Separa em conjunto de testes e de treino
x_train = features[:115, :-1]
y_train = features[:115, -1]

x_test = features[115:, :-1]
y_test = features[115:, -1]

res = np.zeros([22])

for i in range(20):
    # Cria o objeto do classificador
    KNN_model = knn.KNeighborsClassifier(n_neighbors=i + 1)
    KNN_model.fit(x_train, y_train)

    # Faz a predição do conjunto de treino
    y_prediction = KNN_model.predict(x_test)
    res[i] = np.mean(y_prediction == y_test)
    print(i + 1, res[i], np.mean(res))

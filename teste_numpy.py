from sklearn.neural_network import MLPClassifier

#%CARGA DE DADOS

x = ([[0, 0], [0, 1], [1, 0], [1, 1]])
y = ([0, 1, 1, 0])

#%%CONFIG REDE NEURAL 0.000006 relu é a funçao se entrar negativo 0 
mlp = MLPClassifier(verbose=True, max_iter=1000, tol=1e-6, activation='relu')

#%%TREINAMENTO
mlp.fit(x, y)

#%%TESTE
print(mlp.predict([[0, 0]])) # 0
print(mlp.predict([[0, 1]])) # 1
print(mlp.predict([[1, 0]])) # 1    
print(mlp.predict([[1, 1]])) # 0

#%% parametros da rede neural
print("classes = ", mlp.classes_)
print("erro = ", mlp.loss_)
print("amostras visitadas = ", mlp.t_)
print("atributos = ", mlp.n_features_in_)
print("ciclos interacoes n_iter_ = ", mlp.n_iter_)
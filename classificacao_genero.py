#%% BIBLIOTECAS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier  # Certificando que a importação do MLPClassifier está correta
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

#%% CARGA DOS DADOS
df_gen = pd.read_csv('gender_classification_v7.csv')
print('Tabela de dados:\n', df_gen)
input('Aperte uma tecla para continuar: \n')

#%% SELEÇÃO DOS DADOS
# Matriz de atributos
X = df_gen.drop(columns=['gender'])
print("Matriz de entradas:\n", X)
input('Aperte uma tecla para continuar: \n')

# Vetor de classes
y = df_gen['gender']
print("Vetor de classes:\n", y)
input('Aperte uma tecla para continuar: \n')

#%% ONE HOT ENCODER
encoder = OneHotEncoder(sparse_output=False)
X_encoded = encoder.fit_transform(X)

# Convertendo de volta para DataFrame e mantendo os nomes das colunas
X = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(input_features=X.columns))
print("Matriz de entradas codificadas:\n", X)
input('Aperte uma tecla para continuar: \n')

#%% SEPARAÇÃO EM TREINAMENTO E TESTE (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Conjunto de treinamento:", X_train.shape)
print("Conjunto de teste:", X_test.shape)
input('Aperte uma tecla para continuar: \n')

#%% CONFIGURAÇÃO DA REDE (VÁRIOS TESTES COM PARÂMETROS DIFERENTES)
# a) Arquitetura da rede

# a1) Teste com uma camada oculta com 10, 20, 50, 100, 200 neurônios
# b) Função de ativação 'relu' ou 'sigmoid'
# Teste com 'sigmoid' (descomente para testar)
# c) Número de ciclos ou épocas (max_iter)
# Teste com max_iter = 1000, 2000, 3000
# Descomente uma das opções para testar
# d) Taxa de aprendizado
# Teste com 'constant' ou 'adaptive'
# Descomente para testar o 'constant'

# Descomente uma das opções abaixo para testar a rede com diferentes números de neurônios na camada oculta

mlp = MLPClassifier(
    hidden_layer_sizes=(10,), 
    activation='relu', 
    learning_rate='adaptive', 
    max_iter=1000,
    tol=1e-3,
    verbose=True
)

# mlp = MLPClassifier(
#     hidden_layer_sizes=(20,), 
#     activation='relu', 
#     learning_rate='adaptive', 
#     max_iter=2000,
#     tol=1e-3,
#     verbose=True
# )

# mlp = MLPClassifier(
#     hidden_layer_sizes=(50,), 
#     activation='relu', 
#     learning_rate='adaptive', 
#     max_iter=2000,
#     tol=1e-3,
#     verbose=True
# )

# mlp = MLPClassifier(
#     hidden_layer_sizes=(100,), 
#     activation='relu', 
#     learning_rate='adaptive', 
#     max_iter=2000,
#     tol=1e-3,
#     verbose=True
# )

# mlp = MLPClassifier(
#     hidden_layer_sizes=(200,), 
#     activation='relu', 
#     learning_rate='adaptive', 
#     max_iter=2000,
#     tol=1e-3,
#     verbose=True
# )


# a2) Teste com duas camadas ocultas, 10+10, 20+20, 50+50 neurônios
# Descomente uma das opções abaixo para testar a rede com duas camadas ocultas
# b) Função de ativação 'relu' ou 'sigmoid'
# Teste com 'sigmoid' (descomente para testar)
# c) Número de ciclos ou épocas (max_iter)
# Teste com max_iter = 1000, 2000, 3000
# Descomente uma das opções para testar

# d) Taxa de aprendizado
# Teste com 'constant' ou 'adaptive'
# Descomente para testar o 'constant'


# mlp = MLPClassifier(
#     hidden_layer_sizes=(10, 10), 
#     activation='relu', 
#     learning_rate='adaptive', 
#     max_iter=2000,
#     tol=1e-3,
#     verbose=True
# )

# mlp = MLPClassifier(
#     hidden_layer_sizes=(20, 20), 
#     activation='relu', 
#     learning_rate='adaptive', 
#     max_iter=2000,
#     tol=1e-3,
#     verbose=True
# )

# mlp = MLPClassifier(
#     hidden_layer_sizes=(50, 50), 
#     activation='relu', 
#     learning_rate='adaptive', 
#     max_iter=2000,
#     tol=1e-3,
#     verbose=True
# )


# e) Matriz de confusão para cada rodada
# Descomente para treinar a rede e verificar os resultados
mlp.fit(X_train, y_train)
print("\nTreinamento concluído!")

#%% TESTES NO CONJUNTO DE TESTE
y_pred = mlp.predict(X_test)
print("\nPrevisões no conjunto de teste:")
print("Acurácia:", accuracy_score(y_test, y_pred))
print("Matriz de confusão:\n", confusion_matrix(y_test, y_pred))

#%% TESTE COM DADO NÃO VISTO (EXEMPLO NOVO)
# Descomente para testar um novo caso
novo_caso = [1, 13.5, 5.9, 0, 1, 1, 1]  # Exemplo com dados hipotéticos
novo_caso_codificado = encoder.transform([novo_caso])
print("\nNovo caso codificado:", novo_caso_codificado)
previsao = mlp.predict(novo_caso_codificado)
print(f"Novo caso: {novo_caso} = {previsao[0]}")

# %% PARÂMETROS DA REDE NEURAL APÓS O TREINAMENTO
# Descomente para visualizar os parâmetros internos da rede após o treinamento
print("\nParâmetros da rede:")
print("Classes = ", mlp.classes_)  # Lista de classes
print("Erro = ", mlp.loss_)  # Fator de perda (erro)
print("Amostras visitadas = ", mlp.t_)  # Número de amostras de treinamento visitadas
print("Atributos de entrada = ", mlp.n_features_in_)  # Número de atributos de entrada
print("N ciclos = ", mlp.n_iter_)  # Número de iterações no treinamento
print("N de camadas = ", mlp.n_layers_)  # Número de camadas da rede
print("Tamanhos das camadas ocultas: ", mlp.hidden_layer_sizes)
print("N de neurônios de saída = ", mlp.n_outputs_)  # Número de neurônios de saída
print("Função de ativação = ", mlp.out_activation_)  # Função de ativação utilizada
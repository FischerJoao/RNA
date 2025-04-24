#%% BIBLIOTECAS
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

#%% CARGA DOS DADOS
df_gen = pd.read_csv('gender_classification_v7.csv')
print('Tabela de dados:\n', df_gen)
input('Aperte uma tecla para continuar: \n')

#%% SELEÇÃO DOS DADOS
# Matriz de treinamento (atributos)
X = df_gen.drop(columns=['gender'])  # Remove a coluna 'gender' (alvo)
print("Matriz de entradas (treinamento):\n", X)
input('Aperte uma tecla para continuar: \n')

# Vetor de classes (alvo)
y = df_gen['gender']
print("Vetor de classes (treinamento):\n", y)
input('Aperte uma tecla para continuar: \n')

#%% ONE HOT ENCODER - pois os dados são nominais
encoder = OneHotEncoder(sparse_output=False)
X = encoder.fit_transform(X)
print("Matriz de entradas codificadas:\n", X)
input('Aperte uma tecla para continuar: \n')

#%% CONFIGURAÇÃO DA REDE NEURAL
mlp = MLPClassifier(verbose=True, 
                    max_iter=2000, 
                    tol=1e-3, 
                    activation='relu')

#%% TREINAMENTO DA REDE
mlp.fit(X, y)  # Executa o treinamento
print("\nTreinamento concluído!")

#%% TESTES
print('\nTestando os casos de treinamento:')
for caso, classe_real in zip(X, y):
    classe_prevista = mlp.predict([caso])
    print(f'Caso: {caso}, Real: {classe_real}, Previsto: {classe_prevista[0]}')

#%% TESTE DE DADO "NÃO VISTO"
novo_caso = [1, 13.5, 5.9, 0, 0, 0, 0]  # Exemplo de novo caso
novo_caso_codificado = encoder.transform([novo_caso])
print("\nNovo caso codificado:", novo_caso_codificado)

# Previsão
previsao = mlp.predict(novo_caso_codificado)
print(f"Novo caso: {novo_caso} = {previsao[0]}")

#%% ALGUNS PARÂMETROS DA REDE
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
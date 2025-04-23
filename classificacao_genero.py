import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
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

#%% DIVISÃO DOS DADOS EM TREINO E TESTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%% CONFIGURAÇÃO E TREINAMENTO DO MLP
arquiteturas = [
    (10,), (20,), (50,), (100,), (200,),  # Uma camada oculta
    (10, 10), (20, 20), (50, 50)         # Duas camadas ocultas
]
funcoes_ativacao = ['relu', 'logistic']  # 'logistic' é equivalente a 'sigmoid'
taxas_aprendizado = ['constant', 'adaptive']

for arquitetura in arquiteturas:
    for func_ativacao in funcoes_ativacao:
        for taxa_aprendizado in taxas_aprendizado:
            print(f"\nArquitetura: {arquitetura}, Ativação: {func_ativacao}, Taxa: {taxa_aprendizado}")
            
            # Configurar o MLP
            mlp = MLPClassifier(
                hidden_layer_sizes=arquitetura,
                activation=func_ativacao,
                learning_rate=taxa_aprendizado,
                max_iter=1000,
                random_state=42,
                verbose=False
            )
            
            # Treinar o modelo
            mlp.fit(X_train, y_train)
            
            # Avaliar o modelo
            y_pred = mlp.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            
            print(f"Acurácia: {acc:.4f}")
            print("Matriz de Confusão:")
            print(cm)
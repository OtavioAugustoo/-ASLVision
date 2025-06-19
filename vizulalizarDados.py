import pandas as pd
import matplotlib.pyplot as plt
import os

# Carregar dados
df = pd.read_csv("data/asl_dataset.csv")

# Ver as primeiras linhas
print(" Primeiras linhas do dataset:")
print(df.head())

# Ver distribuição de classes
print("\n🔢 Distribuição de classes (letras):")
print(df['label'].value_counts().sort_index())

# Visualizar uma amostra desenhando os landmarks
def plotar_landmarks(index):
    if index >= len(df):
        print(" Índice fora do intervalo.")
        return

    amostra = df.iloc[index]
    x = [amostra[f'x{i}'] for i in range(21)]
    y = [amostra[f'y{i}'] for i in range(21)]

    plt.figure(figsize=(4, 4))
    plt.scatter(x, y, c='blue')
    for i in range(21):
        plt.text(x[i], y[i], str(i), fontsize=8, color='red')
    plt.title(f"Letra: {amostra['label']} (index {index})")
    plt.gca().invert_yaxis()  # inverter eixo Y para parecer mais com a mão real
    plt.grid(True)
    plt.show()

# Exemplo de visualização
while True:
    try:
        idx = int(input("\nDigite o índice da amostra para visualizar (ou -1 para sair): "))
        if idx == -1:
            break
        plotar_landmarks(idx)
    except Exception as e:
        print(f"Erro: {e}")

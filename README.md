# 🤖 HandSignAI

Reconhecimento de sinais do alfabeto ASL  utilizando Visão Computacional e Machine Learning.

---

## 📌 Descrição

**HandSignAI** é um sistema inteligente que reconhece letras do alfabeto ASL a partir de imagens ou vídeo da mão do usuário, utilizando o **MediaPipe** para detecção de landmarks e **RandomForest** (ou outros modelos ML) para classificação dos sinais.

---

## 🧠 Tecnologias utilizadas

- 🖐️ MediaPipe – Extração dos pontos da mão (landmarks)
- 📊 Scikit-learn – Modelos de classificação (Random Forest, LightGBM, etc.)
- 🐍 Python – Linguagem principal
- 🔬 Pandas / NumPy – Manipulação de dados
- 📈 Matplotlib / Seaborn – Visualização de métricas
- 🌐 Flask (opcional) – Interface Web de demonstração

---

## 🗃️ Estrutura do Projeto

```
HandSignAI/
│
├── dataset/
│   └── asl_dataset.csv              # Dataset de treino
├── models/
│   └── modelo_rf.pkl                # Modelo treinado
├── notebooks/
│   └── treino_modelo.ipynb          # Treinamento e avaliação
├── app/
│   └── index.py                     # Aplicação de inferência (opcional Flask)
├── utils/
│   └── preprocess.py                # Pré-processamento dos dados
├── README.md
└── requirements.txt
```

---

## 🚀 Como executar

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/HandSignAI.git
cd HandSignAI
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Execute a aplicação:
```bash
python app/index.py
```

---

## 📊 Exemplo de resultado

```
=== INFERÊNCIA DE EXEMPLO ===
Letra prevista: A
Precisão do modelo: 98.7%
```

---

## ✅ Resultados

| Letra | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
|   A   |   1.00    |  0.98  |   0.99   |
|   I   |   1.00    |  1.00  |   1.00   |
|   O   |   0.98    |  1.00  |   0.99   |

---

## 📚 Futuras melhorias

- Expandir o reconhecimento para palavras inteiras
- Adicionar suporte a webcam em tempo real
- Treinar com mais classes de ASL

---

## 👨‍💻 Autor

Otavio Augusto  
[LinkedIn](https://www.linkedin.com/in/dev-otavio-augusto)  
[GitHub](https://github.com/OtavioAugustoo)

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

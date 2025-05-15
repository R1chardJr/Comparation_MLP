
# Classificação com MLP (Perceptron Multicamadas)

Este projeto utiliza uma rede neural MLP (Multi-Layer Perceptron) para realizar a **classificação binária** de um conjunto de dados tabulares. O modelo é treinado utilizando o Keras (TensorFlow backend) e inclui pré-processamento, avaliação e geração de predições.

## 📁 Estrutura do Projeto

- `Classification_MLP.ipynb`: Notebook principal com todo o pipeline de desenvolvimento.
- Dataset: Arquivo CSV carregado no notebook (não incluído aqui, mas pode ser substituído por outro semelhante).
- Modelo treinado com múltiplas camadas densas e função de ativação `relu` e `sigmoid`.

## 🧠 Modelo

O modelo consiste em:

```python
model = Sequential()
model.add(Dense(125, input_dim=39, activation='relu'))
model.add(Dense(118, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

- Otimizador: `SGD` com taxa de aprendizado ajustável e `momentum`.
- Perda: `binary_crossentropy`
- Métrica: `accuracy`

## 🔧 Funcionalidades

- Pré-processamento com `StandardScaler` e codificação de variáveis categóricas com `get_dummies`.
- Treinamento com validação.
- Geração de gráfico de acurácia e perda.
- Avaliação do modelo com dados de teste.
- Uso do Optuna para ajuste de hiperparâmetros (`momentum`, `learning_rate`, etc.).
- Predição de novas amostras a partir de dados reais ou sintéticos.

## 📊 Visualização

- Correlação entre atributos com `seaborn.heatmap`
- Gráficos de acurácia e perda por época com `matplotlib`

## ▶️ Como Usar

1. Clone o repositório (ou apenas baixe o notebook).
2. Abra o notebook `Classification_MLP.ipynb` no Google Colab ou Jupyter.
3. Substitua ou carregue seu próprio dataset `.csv`.
4. Execute as células sequencialmente para treinar o modelo e fazer predições.

### Exemplo de predição para uma nova amostra

```python
# Usa a primeira linha do dataset como exemplo
amostra = df.iloc[0].drop('target')
amostra_array = np.array(amostra).reshape(1, -1).astype(np.float32)

# Faz a predição
probabilidade = model.predict(amostra_array)
classe_predita = (probabilidade > 0.5).astype(int)
```

## ⚙️ Requisitos

- Python 3.7+
- TensorFlow / Keras
- Pandas, NumPy, Seaborn, Matplotlib
- Scikit-learn
- Optuna (para ajuste de hiperparâmetros)

## 📝 Licença

Este projeto está disponível para uso acadêmico e educacional. Sinta-se à vontade para modificar e experimentar.

## By RichardJr ;)

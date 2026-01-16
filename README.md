# 📈 Previsão de Ações B3 - Engenharia de Machine Learning

> **Status do Projeto:** 🚀 Em desenvolvimento (Refatoração de Experimental para Produção)

Este projeto consiste em uma pipeline completa de Machine Learning para a previsão de preços de fechamento de ações da bolsa brasileira (B3). O diferencial deste repositório é a transição de scripts experimentais do Google Colab para uma **arquitetura de software profissional**, focada em modularização, escalabilidade e boas práticas de Engenharia de Dados.

---

## 🛠 Tecnologias e Ferramentas

* **Linguagem:** Python 3.8+
* **Manipulação de Dados:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn
* **Deep Learning:** Keras, TensorFlow
* **Visualização:** Matplotlib, Seaborn

---

## 🏗 Arquitetura do Sistema

A estrutura foi desenhada utilizando **Programação Orientada a Objetos (POO)** para garantir que cada etapa do processo de dados seja independente e testável.

### Estrutura de Pastas
```text
├── data/               # CSVs originais e processados
├── models/             # Modelos treinados (.pkl ou .h5)
├── src/                # Código-fonte principal
│   ├── config.py       # Configurações e caminhos dinâmicos (Pathlib)
│   ├── data_ingestion.py  # Classe de carga e merge (SQL-like)
│   ├── preprocessing.py   # Limpeza, Normalização e Encoding
│   ├── trainer.py         # Treinamento de Regressão e Redes Neurais
│   └── main.py            # Orquestrador (Entry point)
└── tests/              # Testes unitários
```

### Por que POO e SOLID?
Escalabilidade: Fácil adição de novos modelos sem alterar a ingestão.

Reaproveitamento: O pré-processador é isolado para ser usado em treino ou inferência real.

Manutenibilidade: Separação clara entre lógica de negócio e processamento de dados.

---

## 🔄 Pipeline de Dados
O fluxo segue uma pipeline linear e robusta:

Ingestion: Realiza o merge relacional entre tabelas Fato e Dimensão.

Pre-processing: Trata nulos e aplica MinMaxScaler e LabelEncoder.

Modeling: Treina modelos de Regressão Linear e Redes Neurais Profundas.

Evaluation: Validação via MSE (Mean Squared Error) e R².

---

## 📈 Resultados e Insights
Matriz de Correlação
Identificamos forte correlação entre os valores de abertura (openValueStock) e o alvo de fechamento.

Performance da Rede Neural
O modelo Keras utiliza camadas de Dropout para evitar overfitting e otimizador Adam.

---

## 🚀 Como Rodar o Projeto
1. Clonar e Acessar
```bash
git clone [https://github.com/seu-usuario/b3-stock-prediction.git](https://github.com/seu-usuario/b3-stock-prediction.git)
cd b3-stock-prediction
```
2. Configurar Ambiente Virtual (venv)
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```
3. Instalar Dependências
```bash
pip install -r requirements.txt
```
4. Executar
```bash
python src/main.py
```

---

## 📝 Próximos Passos
[ ] Implementar Docker para containerização.

[ ] Criar uma API com FastAPI para servir o modelo.

[ ] Adicionar suporte a modelos LSTM (Séries Temporais).


---
---
### Desenvolvido por [Linderval Matias] Focado em Engenharia de Dados e Software.
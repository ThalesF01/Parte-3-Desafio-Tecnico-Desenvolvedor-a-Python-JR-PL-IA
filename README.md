# Parte 3: FAQ Bot com TF-IDF + Vizinhos 🤖

Um sistema completo de busca em FAQ usando TF-IDF e algoritmo de vizinhos mais próximos, com interface interativa e métricas detalhadas.

## 📋 Funcionalidades

- **Treinamento automatizado** com `train.py`
- **Bot interativo** via linha de comando
- **Múltiplas respostas** ranqueadas por similaridade
- **Métricas detalhadas** de performance do modelo
- **Testes automatizados** com pytest
- **Salvamento de artefatos** para reutilização

## 🚀 Instalação e Setup

### **Pré-requisitos:**
- Python 3.8+
- pip

### **1. Clone e configure o ambiente:**

```bash
# Clone o repositório
git clone https://github.com/ThalesF01/Parte-3-Desafio-Tecnico-Desenvolvedor-a-Python-JR-PL-IA
cd Parte-3-Desafio-Tecnico-Desenvolvedor-a-Python-JR-PL-IA

# Crie ambiente virtual
python -m venv venv

# Ative o ambiente virtual
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Instale dependências
pip install -r requirements.txt
```

### **2. Dependências necessárias:**

```txt
pandas>=1.3.0
scikit-learn>=1.0.0
pytest>=6.0.0
pickle
```

Se não tiver `requirements.txt`, instale manualmente:
```bash
pip install pandas scikit-learn pytest
```

## 📂 Estrutura do Projeto

```
projeto/
├── data/
│   └── faq.csv              # Dataset com perguntas e respostas
├── artifacts/               # Artefatos gerados pelo treinamento
│   ├── vectorizer.pkl       # TF-IDF vectorizer treinado
│   ├── tfidf_model.pkl     # Modelo de vizinhos mais próximos
│   └── metrics.pkl         # Métricas de performance
├── tests/
│   └── test_train.py       # Testes automatizados
├── train.py                # Script de treinamento
├── faq_bot.py             # Interface interativa do bot
└── requirements.txt        # Dependências
```

## 🔧 Como rodar train.py

### **Executar treinamento:**

```bash
# Treinar o modelo
python train.py
```

**O que acontece:**
1. Carrega dados de `data/faq.csv`
2. Treina vectorizer TF-IDF nas perguntas
3. Treina modelo de vizinhos mais próximos
4. Calcula métricas de performance
5. Salva artefatos em `artifacts/`

### **Output esperado:**
```
Treinamento concluído! Artefatos salvos em artifacts/
Calculando métricas básicas...
Similaridade média: 0.847
Similaridade min/max: 0.234 / 1.000
Acertos exatos (cobertura): 45/50 (90.00%)
Distribuição de similaridades:
  Alta (≥0.8): 38
  Média (0.5-0.8): 10
  Baixa (<0.5): 2
Tamanho do vocabulário: 156
```

## 🤖 Como usar o FAQ Bot

### **Executar bot interativo:**

```bash
# Iniciar o bot (após treinar o modelo)
python faq_bot.py
```

### **Comandos disponíveis:**

#### **1. Pergunta simples:**
```
Pergunta: Para que serve o Python?
Resposta: Python é usado para desenvolvimento web, automação, ciência de dados e IA.
Confiança estimada: 0.92
```

#### **2. Múltiplas respostas:**
```
Pergunta: multi python
Top 3 respostas:
1. Python é usado para desenvolvimento web, automação, ciência de dados e IA. (similaridade: 0.89)
2. Você pode instalar o Python baixando do site oficial python.org e seguindo o instalador. (similaridade: 0.76)
3. Use README.md, docstrings nas funções e comentários claros no código. (similaridade: 0.45)
```

#### **3. Estatísticas do modelo:**
```
Pergunta: stats

=== Estatísticas do Modelo ===
mean_similarity: 0.847
coverage_percent: 90.00
total_questions: 50
vocabulary_size: 156
high_similarity_count: 38
medium_similarity_count: 10
low_similarity_count: 2
```

#### **4. Sair:**
```
Pergunta: sair
```

## 🧪 Executar Testes

### **Todos os testes:**
```bash
python -m pytest -q
```

## 📊 Formato do CSV

O arquivo `data/faq.csv` deve ter a estrutura:

```csv
question,answer
O que é inteligência artificial?,Inteligência artificial é a simulação de processos humanos por máquinas.
Para que serve o Python?,Python é usado para desenvolvimento web, automação, ciência de dados e IA.
Como instalar o Python?,Você pode instalar o Python baixando do site oficial python.org e seguindo o instalador.
```

**Requisitos:**
- Colunas obrigatórias: `question` e `answer`
- Encoding: UTF-8
- Separador: vírgula (`,`)

## 🧠 Explicação Técnica

### **Algoritmo de Busca:**

1. **TF-IDF Vectorization:**
   ```python
   vectorizer = TfidfVectorizer()
   X = vectorizer.fit_transform(questions)
   ```
   - Converte perguntas em vetores numéricos
   - TF-IDF pondera importância das palavras

2. **Vizinhos Mais Próximos:**
   ```python
   nn_model = NearestNeighbors(n_neighbors=1, metric='cosine')
   nn_model.fit(X)
   ```
   - Usa distância cosseno para encontrar similaridade
   - Retorna pergunta mais similar do dataset

3. **Cálculo de Confiança:**
   ```python
   similarity = 1 - cosine_distance
   ```
   - Converte distância em similaridade (0-1)
   - Valores altos = maior confiança

### **Métricas Calculadas:**

| Métrica | Descrição | Interpretação |
|---------|-----------|---------------|
| `mean_similarity` | Similaridade média entre perguntas | Quão bem o modelo diferencia perguntas |
| `coverage_percent` | % de self-matches | Quantas perguntas encontram a si mesmas |
| `high_similarity_count` | Similaridade ≥ 0.8 | Respostas de alta confiança |
| `vocabulary_size` | Palavras únicas | Complexidade do vocabulário |

### **Interpretação dos Resultados:**

- **Similaridade > 0.8**: Resposta confiável ✅
- **Similaridade 0.5-0.8**: Resposta razoável ⚠️
- **Similaridade < 0.5**: Resposta duvidosa ❌

## 🔧 Funcionalidades do Bot

### **1. Busca Simples:**
```python
answer, confidence = ask_question("Como fazer login?")
# Retorna: ("Use email e senha", 0.92)
```

### **2. Busca Múltipla:**
```python
results = ask_multiple("senha", top_k=3)
# Retorna lista com top 3 respostas ranqueadas
```

### **3. Estatísticas:**
```python
stats = get_model_stats()
# Retorna métricas do modelo treinado
```

### **4. Feedback Inteligente:**
- Confiança < 0.5: Sugere reformular pergunta
- Múltiplas opções quando útil
- Interface amigável com comandos intuitivos

## 📈 Exemplos de Uso

### **Cenário 1: FAQ de Sistema**
```csv
question,answer
Como fazer login?,Digite email e senha na tela inicial
Esqueci minha senha,Use a opção "Esqueci senha" no login  
Como alterar perfil?,Acesse Menu > Perfil > Editar
```

**Teste:**
```
Pergunta: login
Resposta: Digite email e senha na tela inicial
Confiança: 0.89
```

### **Cenário 2: Suporte Técnico**  
```csv
question,answer
Sistema lento,Verifique sua conexão de internet
Erro 404,Página não encontrada. Verifique a URL
Erro 500,Erro interno. Tente novamente em alguns minutos
```

**Teste:**
```
Pergunta: multi erro
Top 3 respostas:
1. Erro interno. Tente novamente em alguns minutos (0.87)
2. Página não encontrada. Verifique a URL (0.82)  
3. Verifique sua conexão de internet (0.34)
```

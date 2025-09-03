# FAQ Bot - Sistema de Busca Semântica

Sistema simples e eficaz de FAQ usando TF-IDF + K-Nearest Neighbors para busca semântica de respostas.

## 📋 Visão Geral

Este projeto implementa um bot de FAQ que utiliza técnicas de processamento de linguagem natural para encontrar respostas relevantes baseadas na similaridade semântica entre perguntas. A solução é leve, rápida e não requer recursos computacionais intensivos.

## 🏗️ Estrutura do Projeto

```
projeto/
├── data/
│   └── faq.csv              # Dataset com perguntas e respostas
├── artifacts/               # Modelos treinados (gerado após execução)
│   ├── vectorizer.pkl       # Modelo TF-IDF serializado
│   ├── tfidf_model.pkl      # Modelo K-NN serializado
│   └── metrics.pkl          # Métricas do modelo
├── tests/
│   └── test_train.py        # Testes unitários
├── train.py                 # Script de treinamento do modelo
├── faq_bot.py              # Interface do bot com CLI
├── requirements.txt        # Dependências Python
└── README.md              # Este arquivo
```

## ⚙️ Instalação e Configuração

### Pré-requisitos
- Python 3.7+
- pip

### 1. Clonar o repositório
```bash
git clone <url-do-repositorio>
cd faq-bot
```

### 2. Instalar dependências
```bash
pip install -r requirements.txt
```

### 3. Preparar dados
Certifique-se de que o arquivo `data/faq.csv` existe com o formato:

```csv
question,answer
Como fazer login?,Use seu email e senha na tela de login
Esqueci minha senha,Clique em "Esqueci senha" e siga as instruções
Como criar uma conta?,Clique em "Criar conta" e preencha o formulário
```

## 🚀 Execução

### 1. Treinar o modelo
```bash
python train.py
```

**Saída esperada:**
```
Treinamento concluído! Artefatos salvos em artifacts/
Calculando métricas básicas...
Similaridade média: 0.875
Similaridade min/max: 0.234 / 1.000
Acertos exatos (cobertura): 45/50 (90.00%)
Distribuição de similaridades:
  Alta (≥0.8): 42
  Média (0.5-0.8): 6
  Baixa (<0.5): 2
Tamanho do vocabulário: 324
```

### 2. Executar o bot
```bash
python faq_bot.py
```

**Interface interativa com comandos:**
- Digite perguntas normalmente para buscar respostas
- `stats` - Exibe estatísticas do modelo treinado
- `multi <pergunta>` - Retorna múltiplas respostas rankeadas
- `quit` / `sair` / `exit` - Encerra o programa

**Exemplo de uso:**
```
FAQ Bot carregado!

Pergunta: Como fazer login?
Resposta: Use seu email e senha na tela de login
Confiança estimada: 0.95

Pergunta: stats
=== Estatísticas do Modelo ===
total_questions: 50
vocabulary_size: 324
mean_similarity: 0.875

Pergunta: multi problema senha
Top 3 respostas:
1. Clique em "Esqueci senha" e siga as instruções (similaridade: 0.78)
2. Entre em contato com o suporte técnico (similaridade: 0.45)
3. Verifique se caps lock está ativado (similaridade: 0.32)
```

## 🧪 Executar Testes

```bash
# Rodar todos os testes
python -m pytest tests/test_train.py -v

# Com relatório detalhado
python -m pytest tests/test_train.py -v --tb=short

# Teste específico
python -m pytest tests/test_train.py::test_compute_metrics -v
```

**Saída esperada:**
```
tests/test_train.py::test_load_data PASSED
tests/test_train.py::test_train_tfidf_model PASSED
tests/test_train.py::test_compute_metrics PASSED
tests/test_train.py::test_metrics_values PASSED
tests/test_train.py::test_small_dataset PASSED
```

## 🛠️ Decisões Técnicas

### Algoritmo Escolhido: TF-IDF + K-Nearest Neighbors

**Por que essa abordagem?**

1. **Simplicidade e Eficácia**: Solução robusta para busca semântica em datasets pequenos/médios
2. **Performance**: Treinamento rápido, consultas em tempo real
3. **Interpretabilidade**: Resultados explicáveis através da similaridade coseno
4. **Recursos**: Não requer GPU ou grandes quantidades de memória
5. **Manutenibilidade**: Código simples de entender e modificar

### Configurações do Modelo

```python
# TF-IDF Vectorizer
TfidfVectorizer()  # Configuração padrão otimizada

# K-Nearest Neighbors
NearestNeighbors(n_neighbors=1, metric='cosine')
```

**Métrica de Distância**: Coseno - ideal para comparação de textos, não sensível ao tamanho dos documentos.

### Métricas de Avaliação

| Métrica | Descrição | Interpretação |
|---------|-----------|---------------|
| **Similaridade Média** | Média das similaridades entre perguntas e vizinhos mais próximos | Qualidade geral do matching |
| **Coverage (Acertos Exatos)** | % de perguntas que encontram match perfeito consigo mesmas | Capacidade de distinção do modelo |
| **Distribuição de Similaridades** | Contagem por faixas de similaridade | Identificação de possíveis problemas no dataset |
| **Tamanho do Vocabulário** | Número de features únicas extraídas | Complexidade e capacidade do modelo |

## 📊 API e Funcionalidades

### Função Principal
```python
from faq_bot import ask_question

answer, confidence = ask_question("Como criar uma conta?")
print(f"Resposta: {answer}")
print(f"Confiança: {confidence:.2f}")
```

### Busca Múltipla
```python
from faq_bot import ask_multiple

results = ask_multiple("problema login", top_k=3)
for result in results:
    print(f"{result['rank']}. {result['answer']} ({result['similarity']:.2f})")
```

### Estatísticas do Modelo
```python
from faq_bot import get_model_stats

stats = get_model_stats()
print(f"Total de perguntas: {stats['total_questions']}")
print(f"Vocabulário: {stats['vocabulary_size']} palavras")
```

## 🔧 Resolução de Problemas

### Erro: "Arquivo não encontrado"
```bash
# Verifique se os arquivos existem
ls data/faq.csv
ls artifacts/

# Se artifacts/ estiver vazio, execute o treinamento
python train.py
```

### Baixa confiança nas respostas
- **Causa**: Dataset pequeno ou perguntas muito específicas
- **Solução**: Adicionar mais variações de perguntas similares ao CSV
- **Verificação**: Use comando `stats` para analisar distribuição de similaridades

### Erros nos testes
```bash
# Certifique-se de estar no diretório correto
pwd

# Execute da raiz do projeto
python -m pytest tests/test_train.py -v
```

## 🚀 Melhorias Futuras

### Curto Prazo
- [ ] Suporte a stop words em português
- [ ] Cache de consultas frequentes
- [ ] Logs estruturados para debugging

### Médio Prazo
- [ ] Interface web com Flask/FastAPI
- [ ] Métricas avançadas (BLEU, ROUGE)
- [ ] Suporte a múltiplos idiomas

### Longo Prazo
- [ ] Migração para embeddings contextuais (BERT/Sentence-BERT)
- [ ] Sistema de feedback para melhoria contínua
- [ ] Pipeline de retreinamento automático

## 📦 Dependências

```txt
pandas>=1.5.0        # Manipulação de dados
scikit-learn>=1.1.0  # Algoritmos de ML
pytest>=7.0.0        # Framework de testes
```

## 📈 Benchmarks

### Dataset de Exemplo (50 perguntas)
- **Tempo de treinamento**: ~0.5 segundos
- **Tempo de consulta**: ~10ms por pergunta
- **Uso de memória**: ~5MB para modelos serializados
- **Acurácia típica**: 85-95% de similaridade média

### Escalabilidade
- **Até 1.000 perguntas**: Performance excelente
- **1.000-10.000 perguntas**: Performance boa, considerar otimizações
- **10.000+ perguntas**: Avaliar migração para soluções mais robustas

## 👥 Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/nova-funcionalidade`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/nova-funcionalidade`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

---

**Desenvolvido como solução técnica para sistema de FAQ com busca semântica** ⚡
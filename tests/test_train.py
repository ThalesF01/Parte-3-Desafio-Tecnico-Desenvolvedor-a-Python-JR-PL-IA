import pytest
from pathlib import Path
import pandas as pd
from train import load_data, train_tfidf_model, compute_metrics

DATA_PATH = Path("data/faq.csv")

def test_load_data():
    df = load_data(DATA_PATH)
    assert not df.empty
    assert "question" in df.columns
    assert "answer" in df.columns

def test_train_tfidf_model():
    df = load_data(DATA_PATH)
    vectorizer, nn_model = train_tfidf_model(df)  # passar o DataFrame completo
    assert vectorizer is not None
    assert nn_model is not None
    X = vectorizer.transform(df['question'])
    assert X.shape[0] == len(df)

def test_compute_metrics():
    """Testa se compute_metrics retorna métricas corretas"""
    df = load_data(DATA_PATH)
    vectorizer, nn_model = train_tfidf_model(df)
    metrics = compute_metrics(df, vectorizer, nn_model)
    
    # Verifica se retorna dicionário com métricas esperadas
    assert isinstance(metrics, dict)
    assert 'mean_similarity' in metrics
    assert 'coverage_percent' in metrics
    assert 'total_questions' in metrics
    assert metrics['total_questions'] == len(df)

def test_metrics_values():
    """Testa se os valores das métricas fazem sentido"""
    df = load_data(DATA_PATH)
    vectorizer, nn_model = train_tfidf_model(df)
    metrics = compute_metrics(df, vectorizer, nn_model)
    
    # Similaridade deve estar entre 0 e 1
    assert 0 <= metrics['mean_similarity'] <= 1
    assert 0 <= metrics['min_similarity'] <= 1
    assert 0 <= metrics['max_similarity'] <= 1
    
    # Coverage deve estar entre 0 e 100
    assert 0 <= metrics['coverage_percent'] <= 100
    
    # Contadores devem somar o total
    total_sim = (metrics['high_similarity_count'] + 
                metrics['medium_similarity_count'] + 
                metrics['low_similarity_count'])
    assert total_sim == metrics['total_questions']

def test_small_dataset():
    """Testa com dataset pequeno"""
    small_df = pd.DataFrame({
        'question': ['Como fazer login?', 'Esqueci senha', 'Criar conta'],
        'answer': ['Use email e senha', 'Clique esqueci senha', 'Clique criar conta']
    })
    
    vectorizer, nn_model = train_tfidf_model(small_df)
    metrics = compute_metrics(small_df, vectorizer, nn_model)
    
    assert metrics['total_questions'] == 3
    assert metrics['vocabulary_size'] > 0
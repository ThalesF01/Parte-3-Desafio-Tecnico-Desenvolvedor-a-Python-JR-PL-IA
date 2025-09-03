import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, Dict
import os

def load_data(csv_path: str) -> pd.DataFrame:
    """Carrega dados do CSV"""
    return pd.read_csv(csv_path)

def train_tfidf_model(data: pd.DataFrame) -> Tuple[TfidfVectorizer, NearestNeighbors]:
    """
    Treina o TF-IDF e o modelo de vizinhos
    """
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['question'])
    nn_model = NearestNeighbors(n_neighbors=1, metric='cosine')
    nn_model.fit(X)
    return vectorizer, nn_model

def compute_metrics(data: pd.DataFrame, vectorizer: TfidfVectorizer, nn_model: NearestNeighbors) -> Dict:
    """
    Calcula métricas básicas:
    - Similaridade média entre perguntas e vizinhos mais próximos
    - Quantidade de acertos exatos (própria pergunta como vizinho)
    - Distribuição de similaridades
    """
    X = vectorizer.transform(data['question'])
    distances, indices = nn_model.kneighbors(X)
    
    # Similaridade coseno é 1 - distância
    similarities = 1 - distances.flatten()
    mean_similarity = similarities.mean()
    min_similarity = similarities.min()
    max_similarity = similarities.max()
    
    # Quantidade de acertos exatos (índice do vizinho == índice da pergunta)
    exact_matches = sum(indices.flatten() == list(range(len(data))))
    coverage_percent = (exact_matches / len(data)) * 100
    
    # Contar por faixas de similaridade
    high_sim = sum(similarities >= 0.8)
    med_sim = sum((similarities >= 0.5) & (similarities < 0.8))
    low_sim = sum(similarities < 0.5)
    
    metrics = {
        'mean_similarity': mean_similarity,
        'min_similarity': min_similarity,
        'max_similarity': max_similarity,
        'exact_matches': exact_matches,
        'coverage_percent': coverage_percent,
        'high_similarity_count': high_sim,
        'medium_similarity_count': med_sim,
        'low_similarity_count': low_sim,
        'total_questions': len(data),
        'vocabulary_size': len(vectorizer.vocabulary_)
    }
    
    # Print das métricas
    print(f"Similaridade média: {mean_similarity:.3f}")
    print(f"Similaridade min/max: {min_similarity:.3f} / {max_similarity:.3f}")
    print(f"Acertos exatos (cobertura): {exact_matches}/{len(data)} ({coverage_percent:.2f}%)")
    print(f"Distribuição de similaridades:")
    print(f"  Alta (≥0.8): {high_sim}")
    print(f"  Média (0.5-0.8): {med_sim}")
    print(f"  Baixa (<0.5): {low_sim}")
    print(f"Tamanho do vocabulário: {len(vectorizer.vocabulary_)}")
    
    return metrics

if __name__ == "__main__":
    df = load_data("data/faq.csv")
    vectorizer, nn_model = train_tfidf_model(df)

    os.makedirs("artifacts", exist_ok=True)

    with open("artifacts/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    with open("artifacts/tfidf_model.pkl", "wb") as f:
        pickle.dump(nn_model, f)

    print("Treinamento concluído! Artefatos salvos em artifacts/")
    print("Calculando métricas básicas...")
    metrics = compute_metrics(df, vectorizer, nn_model)
    
    # Salvar métricas também
    with open("artifacts/metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, List, Dict

# carregar artefatos
with open("artifacts/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("artifacts/tfidf_model.pkl", "rb") as f:
    nn_model = pickle.load(f)

# carregar FAQ
faq = pd.read_csv("data/faq.csv")

def ask_question(question: str) -> Tuple[str, float]:
    """
    Retorna a resposta mais próxima e a confiança (similaridade coseno)
    """
    q_vec = vectorizer.transform([question])
    distances, indices = nn_model.kneighbors(q_vec)
    index = indices[0][0]
    similarity = 1 - distances[0][0]  # similaridade coseno
    answer = faq.iloc[index]['answer']
    return answer, similarity

def ask_multiple(question: str, top_k: int = 3) -> List[Dict]:
    """
    Retorna múltiplas respostas ordenadas por similaridade
    """
    q_vec = vectorizer.transform([question])
    # Ajustar k para não exceder o número de perguntas
    k = min(top_k, len(faq))
    distances, indices = nn_model.kneighbors(q_vec, n_neighbors=k)
    
    results = []
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        results.append({
            'rank': i + 1,
            'answer': faq.iloc[idx]['answer'],
            'question': faq.iloc[idx]['question'],
            'similarity': 1 - dist
        })
    
    return results

def get_model_stats() -> Dict:
    """Retorna estatísticas básicas do modelo"""
    try:
        with open("artifacts/metrics.pkl", "rb") as f:
            metrics = pickle.load(f)
        return metrics
    except FileNotFoundError:
        return {
            'total_questions': len(faq),
            'vocabulary_size': len(vectorizer.vocabulary_)
        }

if __name__ == "__main__":
    print("FAQ Bot carregado!")
    print("Digite 'stats' para ver estatísticas, 'multi <pergunta>' para múltiplas respostas")
    print()
    
    while True:
        user_input = input("Pergunta: ")
        
        if user_input.lower() in {"sair", "exit", "quit"}:
            break
        
        elif user_input.lower() == "stats":
            stats = get_model_stats()
            print("\n=== Estatísticas do Modelo ===")
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.3f}")
                else:
                    print(f"{key}: {value}")
            print()
            continue
        
        elif user_input.lower().startswith("multi "):
            question = user_input[6:]  # remove "multi "
            results = ask_multiple(question)
            print(f"\nTop {len(results)} respostas:")
            for r in results:
                print(f"{r['rank']}. {r['answer']} (similaridade: {r['similarity']:.2f})")
            print()
            continue
        
        answer, confidence = ask_question(user_input)
        print(f"Resposta: {answer}")
        print(f"Confiança estimada: {confidence:.2f}")  # valor entre 0 e 1
        
        # Sugestão se confiança baixa
        if confidence < 0.5:
            print("Dica: Confiança baixa. Tente reformular a pergunta.")
        print()
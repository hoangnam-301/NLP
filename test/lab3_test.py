import sys
import os

# Them duong dan src vao he thong
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.representations.word_embedder import WordEmbedder

def main():
    try:
        # Load model 50-dimensional trained on Wikipedia
        embedder = WordEmbedder('glove-wiki-gigaword-50')
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 1. Get vector for 'king'
    king_vec = embedder.get_vector('king')
    if king_vec is not None:
        print(f"\nVector for 'king' (first 5 elements): {king_vec[:5]}")
    else:
        print("\nWord 'king' not in vocabulary.")

    # 2. Similarity operations
    sim_king_queen = embedder.get_similarity('king', 'queen')
    sim_king_man = embedder.get_similarity('king', 'man')
    print(f"Similarity (king, queen): {sim_king_queen:.4f}")
    print(f"Similarity (king, man): {sim_king_man:.4f}")

    # 3. Most similar to 'computer'
    most_sim = embedder.get_most_similar('computer', top_n=10)
    print(f"\nTop 10 most similar to 'computer':")
    for word, score in most_sim:
        print(f" - {word}: {score:.4f}")

    # 4. Document embedding
    doc = "The queen rules the country."
    doc_vec = embedder.embed_document(doc)
    print(f"\nDocument vector for sentence: '{doc}'")
    print(f"(first 5 elements): {doc_vec[:5]}")

if __name__ == "__main__":
    main()
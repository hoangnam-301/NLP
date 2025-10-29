import os, sys
sys.stdout.reconfigure(encoding="utf-8")

# Thêm thư mục gốc vào sys.path để import được src/
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.preprocessing.simple_tokenizer import SimpleTokenizer
from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.core.dataset_loaders import load_raw_text_data

def print_tokenization(title: str, text: str, simple_tok, regex_tok, n_preview: int = 20):
    print(f"\n=== {title} ===")
    print(f"Input: {text}")
    s = simple_tok.tokenize(text)
    r = regex_tok.tokenize(text)
    print(f"SimpleTokenizer -> {s[:n_preview]}")
    print(f"RegexTokenizer  -> {r[:n_preview]}")

def main():
    simple_tok = SimpleTokenizer()
    regex_tok = RegexTokenizer()

    # --- Evaluation trên 3 câu mẫu ---
    sentences = [
        "Hello, world! This is a test.",
        "NLP is fascinating... isn't it?",
        "Let's see how it handles 123 numbers and punctuation!"
    ]
    for i, sent in enumerate(sentences, start=1):
        print_tokenization(f"Sentence {i}", sent, simple_tok, regex_tok)

    # --- Tokenization trên dataset UD_English-EWT ---
    print("\n--- Tokenizing Sample Text from UD_English-EWT ---")
    dataset_path = os.path.join(os.path.dirname(__file__), "UD_English-EWT", "en_ewt-ud-train.txt")

    if not os.path.exists(dataset_path):
        print(f"Không tìm thấy file dataset: {dataset_path}")
        print("→ Hãy chắc rằng file 'en_ewt-ud-train.txt' nằm trong thư mục 'NLP/UD_English-EWT/'")
        return

    raw_text = load_raw_text_data(dataset_path)
    sample_text = raw_text[:500]  # lấy 500 ký tự đầu tiên

    print(f"\nOriginal Sample (first 100 chars): {sample_text[:100]}...")
    simple_tokens = simple_tok.tokenize(sample_text)
    regex_tokens  = regex_tok.tokenize(sample_text)
    print(f"\nSimpleTokenizer Output (first 20 tokens): {simple_tokens[:20]}")
    print(f"RegexTokenizer  Output (first 20 tokens): {regex_tokens[:20]}")
    print(f"\nToken counts → Simple: {len(simple_tokens)}, Regex: {len(regex_tokens)}")

if __name__ == "__main__":
    main()

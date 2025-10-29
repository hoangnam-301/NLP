# src/core/dataset_loaders.py
def load_raw_text_data(path: str) -> str:
    """Đọc toàn bộ file văn bản thô và trả về một chuỗi lớn."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

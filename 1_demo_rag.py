# 1_demo_rag.py - RAG tối giản bằng thư viện chuẩn Python
# Luồng: Ingestion -> Retrieval (TF-IDF + cosine) -> Augment (ghép prompt) -> Generate (mock)
import math, re, sys
from collections import Counter

# Đảm bảo hiển thị đúng tiếng Việt trên terminal Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DOCS = [
    {"id": "doc1", "text": "RAG là kỹ thuật kết hợp truy xuất thông tin và sinh văn bản để giảm ảo giác của LLM."},
    {"id": "doc2", "text": "NLQ cho phép người dùng hỏi dữ liệu bằng ngôn ngữ tự nhiên thay vì viết SQL."},
    {"id": "doc3", "text": "Cơ sở dữ liệu vector lưu embedding của các đoạn văn bản để tìm kiếm theo ngữ nghĩa trong RAG."},
    {"id": "doc4", "text": "Fine-tuning mô hình rất tốn kém, trong khi RAG chỉ cần cập nhật kho tài liệu."},
]

STOPWORDS = {"là", "và", "của", "các", "cho", "để", "thay", "vì", "trong", "khi", "gì",
             "tại", "sao", "dùng", "một", "những", "được", "bằng", "chỉ", "cần"}

def tokenize(text):
    return [w for w in re.findall(r"\w+", text.lower()) if w not in STOPWORDS]

# ---- 1. Ingestion: IDF + vector TF-IDF cho từng đoạn ----
N = len(DOCS)
doc_tokens = [tokenize(d["text"]) for d in DOCS]
df = Counter(w for toks in doc_tokens for w in set(toks))
idf = {w: math.log((1 + N) / (1 + c)) + 1 for w, c in df.items()}

def vectorize(tokens):
    tf = Counter(tokens)
    return {w: tf[w] * idf[w] for w in tf if w in idf}

doc_vecs = [vectorize(t) for t in doc_tokens]

def cosine(a, b):
    dot = sum(a[w] * b.get(w, 0) for w in a)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0

# ---- 2. Retrieval ----
def retrieve(query, top_k=2, min_score=0.15):
    q = vectorize(tokenize(query))
    scored = sorted(((cosine(q, v), d) for v, d in zip(doc_vecs, DOCS)),
                    key=lambda x: x[0], reverse=True)
    return [(s, d) for s, d in scored[:top_k] if s >= min_score]

# ---- 3. Augment ----
def build_prompt(query, hits):
    context = "\n".join(f"[{d['id']}] {d['text']}" for _, d in hits)
    return ("Chỉ trả lời dựa trên TÀI LIỆU dưới đây. Nếu không có thông tin, hãy nói không biết. "
            "Ghi nguồn dạng [docX].\n\n"
            f"TÀI LIỆU:\n{context}\n\nCÂU HỎI: {query}")

# ---- 4. Generate (giả lập; thực tế gửi prompt cho LLM) ----
def generate(query, hits):
    if not hits:
        return "Tôi không tìm thấy thông tin này trong tài liệu."
    best = hits[0][1]
    return f"{best['text']} [{best['id']}]"

if __name__ == "__main__":
    for q in ["RAG là kỹ thuật gì?",
              "Vì sao dùng RAG thay vì fine-tuning?",
              "Thủ đô của nước Pháp là gì?",       # ngoài tài liệu -> từ chối
              "Làm sao để AI bớt bịa đặt?"]:       # cùng nghĩa doc1 nhưng khác từ -> TF-IDF bó tay
        hits = retrieve(q)
        print("=" * 60)
        print("Câu hỏi:", q)
        for s, d in hits:
            print(f"  - [{d['id']}] điểm {s:.2f}")
        print("Prompt gửi LLM:\n" + build_prompt(q, hits))
        print("Trả lời (mock):", generate(q, hits))

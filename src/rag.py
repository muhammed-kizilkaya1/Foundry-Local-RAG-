import sqlite3
import json
import numpy as np
from openai import OpenAI
import config

_chunk_cache = None


def load_chunks_to_memory(force_reload=False):
    global _chunk_cache
    if _chunk_cache is not None and not force_reload:
        return _chunk_cache

    try:
        conn = sqlite3.connect(config.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT text, embedding, source, page FROM chunks")
        rows = cursor.fetchall()
        conn.close()
    except sqlite3.OperationalError:
        _chunk_cache = []
        return _chunk_cache

    cache = []
    for text, emb_json, source, page in rows:
        emb_np = np.array(json.loads(emb_json), dtype=np.float32)
        cache.append({"text": text, "embedding": emb_np, "source": source, "page": page})

    _chunk_cache = cache
    return _chunk_cache


def get_chunk_count():
    return len(load_chunks_to_memory())


def cosine_similarity_np(a, b):
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


class EmbeddingServerError(RuntimeError):
    pass


def _get_query_embedding(client, query):
    try:
        response = client.embeddings.create(model=config.EMBEDDING_MODEL, input=[query])
    except Exception as e:
        raise EmbeddingServerError(
            f"Embedding sunucusuna ulaşılamadı ({config.API_BASE_URL}). "
            f"Yerel model sunucunuzun çalıştığından emin olun."
        ) from e
    return np.array(response.data[0].embedding, dtype=np.float32)


def get_top_chunks_with_scores(query, top_k=None):
    top_k = top_k or config.TOP_K
    client = OpenAI(base_url=config.API_BASE_URL, api_key=config.API_KEY)
    query_emb = _get_query_embedding(client, query)

    cache = load_chunks_to_memory()
    if not cache:
        return []

    results = []
    for item in cache:
        score = cosine_similarity_np(query_emb, item["embedding"])
        results.append({
            "text": item["text"],
            "source": item["source"],
            "page": item["page"],
            "score": score,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def get_top_chunks(query, top_k=None):
    return [r["text"] for r in get_top_chunks_with_scores(query, top_k)]


def answer_query(query, top_k=None):
    top_k = top_k or config.TOP_K
    print("🔍 İlgili belgeler aranıyor...")

    try:
        results = get_top_chunks_with_scores(query, top_k=top_k)
    except EmbeddingServerError as e:
        return f"⚠️ {e}", []

    # Eşik kontrolü
    if not results or results[0]["score"] < config.RELEVANCE_THRESHOLD:
        return "Bu bilgi notlarımda yok.", results

    context = "\n\n---\n\n".join(r["text"] for r in results)

    # 🔽 Sistem promptu daha keskin ve net
    system_prompt = f"""Sen bir {config.SUBJECT_NAME} öğretmenisin ve TÜRKÇE cevap veriyorsun.
Kullanıcının sorusunu SADECE ve SADECE aşağıda verilen bağlama (Context) göre cevapla.

KURALLAR:
1. Bağlamda cevap yoksa, kesinlikle tahmin etme, uydurma veya kendi bilgini kullanma.
2. Bağlamda cevap yoksa, TEK BİR CÜMLE ile sadece "Bu bilgi notlarımda yok." de. Başka hiçbir şey söyleme.
3. Bağlamda cevap varsa, onu aynen kullan, yorum yapma, ek bilgi ekleme.

Bağlam:
{context}

Soru: {query}

Cevap (sadece bağlamdan al, yoksa 'Bu bilgi notlarımda yok.' yaz):"""

    user_prompt = f"""Bağlam:
{context}

Soru: {query}

Cevap (sadece bağlamdan al, yoksa 'Bu bilgi notlarımda yok.' yaz):"""

    print("🧠 Cevap oluşturuluyor...")
    try:
        client = OpenAI(base_url=config.API_BASE_URL, api_key=config.API_KEY)
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        answer = response.choices[0].message.content
        # Son bir filtre: Eğer cevap çok kısa veya anlamsızsa (isteğe bağlı)
        # if len(answer.split()) < 3 and answer not in ["Bu bilgi notlarımda yok."]:
        #     answer = "Bu bilgi notlarımda yok."
    except Exception as e:
        answer = (
            f"⚠️ Dil modeli sunucusuna ulaşılamadı ({config.API_BASE_URL}). "
            f"Yerel model sunucunuzun çalıştığından emin olun.\n\nDetay: {e}"
        )

    return answer, results


def main():
    print(f"🤖 {config.SUBJECT_NAME} Asistanı (Tamamen Çevrimdışı)")
    print("=" * 45)
    print(f"{config.SUBJECT_NAME} notları hakkında soru sorun.")
    print("Çıkmak için 'quit' veya 'exit' yazın.\n")
    while True:
        query = input("❓ Siz: ").strip()
        if not query:
            continue
        if query.lower() in ['quit', 'exit', 'q']:
            print("👋 Görüşmek üzere!")
            break
        answer, _ = answer_query(query)
        print(f"\n💬 Asistan: {answer}\n")
        print("-" * 45)


if __name__ == "__main__":
    main()

from rag import get_top_chunks_with_scores, EmbeddingServerError
import config


def main():
    query = input("🔍 Arama sorgunuzu girin: ").strip()
    if not query:
        return

    try:
        results = get_top_chunks_with_scores(query, top_k=3)
    except EmbeddingServerError as e:
        print(f"⚠️ {e}")
        return

    if not results:
        print("⚠️ Sonuç bulunamadı. Önce 'ingest.py' çalıştırdınız mı?")
        return

    print(f"\n📊 '{query}' için en iyi sonuçlar:")
    for i, r in enumerate(results):
        page_info = f", s. {r['page']}" if r.get("page") else ""
        print(f"\n--- Sonuç {i+1} (skor: {r['score']:.4f}) — {r['source']}{page_info} ---")
        print(r["text"])


if __name__ == "__main__":
    main()

import sqlite3
import json
import os
from openai import OpenAI
from tqdm import tqdm
from pypdf import PdfReader
import config


def recursive_split_text(text, chunk_size=None, overlap=None, separators=("\n\n", "\n", " ", "")):
    """Metni chunk_size'a yakın parçalara böler, mümkünse doğal ayraçlardan keser.
    NOT: start_idx her turda en az 1 karakter ilerler; aksi halde bazı metin
    desenlerinde (ayraç start_idx'e çok yakınsa) sonsuz döngüye girebiliyordu."""
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP

    if not text or len(text) <= chunk_size:
        return [text.strip()] if text and text.strip() else []

    chunks = []
    start_idx = 0
    text_len = len(text)

    while start_idx < text_len:
        end_idx = start_idx + chunk_size
        if end_idx >= text_len:
            tail = text[start_idx:].strip()
            if tail:
                chunks.append(tail)
            break

        best_sep_idx = -1
        best_sep_priority = -1
        for priority, sep in enumerate(separators):
            if sep == "":
                continue
            idx = text.rfind(sep, start_idx, end_idx)
            if idx != -1 and priority > best_sep_priority:
                best_sep_priority = priority
                best_sep_idx = idx

        if best_sep_idx != -1:
            end = best_sep_idx + len(separators[best_sep_priority])
        else:
            end = end_idx

        chunk = text[start_idx:end].strip()
        if chunk:
            chunks.append(chunk)

        next_start = end - overlap
        if next_start <= start_idx:
            next_start = end
        start_idx = next_start

        if start_idx >= text_len - 10:
            break

    return [c for c in chunks if c]


def read_document(filepath):
    """Dosyayı okur ve [(sayfa_no, metin), ...] listesi döner.
    .txt dosyaları tek 'sayfa' (1) olarak kabul edilir."""
    if filepath.endswith(".txt"):
        with open(filepath, "r", encoding="utf-8") as f:
            return [(1, f.read())]
    elif filepath.endswith(".pdf"):
        reader = PdfReader(filepath)
        pages = []
        for page_num, page in enumerate(reader.pages, start=1):
            extracted = page.extract_text()
            if extracted and extracted.strip():
                pages.append((page_num, extracted))
        return pages
    return None


def main():
    if not os.path.exists(config.DATA_DIR):
        os.makedirs(config.DATA_DIR)

    if not os.path.exists(config.DOC_FOLDER):
        os.makedirs(config.DOC_FOLDER)
        print(f"📁 '{config.DOC_FOLDER}' klasörü oluşturuldu. Lütfen {config.SUBJECT_NAME} .txt/.pdf dosyalarını buraya ekleyin.")
        return

    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            page INTEGER,
            text TEXT,
            embedding TEXT,
            chunk_index INTEGER
        )
    ''')
    try:
        cursor.execute("ALTER TABLE chunks ADD COLUMN page INTEGER")
    except sqlite3.OperationalError:
        pass

    cursor.execute("DELETE FROM chunks")
    conn.commit()
    print("🧹 Eski veriler temizlendi, tablo hazır.")

    all_files = [f for f in os.listdir(config.DOC_FOLDER) if f.endswith((".txt", ".pdf"))]
    if not all_files:
        print(f"⚠️ {config.DOC_FOLDER} içinde .txt veya .pdf dosyası bulunamadı.")
        conn.close()
        return

    print(f"📄 {len(all_files)} belge bulundu. İşleniyor...")
    all_chunks = []
    for filename in all_files:
        filepath = os.path.join(config.DOC_FOLDER, filename)
        try:
            pages = read_document(filepath)
        except Exception as e:
            print(f"  ⚠️ {filename} okunamadı: {e}")
            continue

        if not pages:
            print(f"  ⚠️ {filename} atlanıyor (boş veya okunamadı).")
            continue

        for page_num, page_text in pages:
            chunks = recursive_split_text(page_text)
            for idx, chunk in enumerate(chunks):
                if chunk.strip():
                    all_chunks.append({
                        "source": filename,
                        "page": page_num,
                        "text": chunk.strip(),
                        "chunk_index": idx,
                    })

    if not all_chunks:
        print("⚠️ Hiç parça (chunk) üretilemedi. Dosyaları kontrol edin.")
        conn.close()
        return

    try:
        client = OpenAI(base_url=config.API_BASE_URL, api_key=config.API_KEY)
        texts = [c["text"] for c in all_chunks]

        print("🔮 Embedding'ler oluşturuluyor... (biraz sürebilir)")
        batch_size = 50
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i + batch_size]
            response = client.embeddings.create(model=config.EMBEDDING_MODEL, input=batch)
            all_embeddings.extend([data.embedding for data in response.data])
    except Exception as e:
        print(f"❌ Embedding sunucusuna bağlanılamadı ({config.API_BASE_URL}).")
        print(f"   Yerel model sunucunuzun çalıştığından emin olun. Hata: {e}")
        conn.close()
        return

    print("💾 SQLite'a kaydediliyor...")
    for chunk, embedding in tqdm(zip(all_chunks, all_embeddings), total=len(all_chunks)):
        embedding_json = json.dumps(embedding)
        cursor.execute(
            "INSERT INTO chunks (source, page, text, embedding, chunk_index) VALUES (?, ?, ?, ?, ?)",
            (chunk["source"], chunk["page"], chunk["text"], embedding_json, chunk["chunk_index"]),
        )
    conn.commit()
    print(f"✅ {len(all_chunks)} parça başarıyla eklendi.")
    conn.close()


if __name__ == "__main__":
    main()

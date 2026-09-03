
# 📚 Yerel RAG Asistanı — Çevrimdışı Belge Sorgulama

Bu proje, **tamamen yerel** çalışan bir Soru-Cevap asistanıdır. Kendi belgelerinizi (ders notları, kılavuzlar, makaleler) yükleyip, onlar üzerinden doğal dilde sorular sorabilirsiniz. Tüm işlemler **internet bağlantısı olmadan**, kendi bilgisayarınızda gerçekleşir.

Proje, **RAG (Retrieval-Augmented Generation)** desenini kullanır: sorunuzu önce belge parçalarıyla eşleştirir, ardından bu parçaları bir dil modeline vererek doğru ve kaynak gösteren cevaplar üretir.

---

## 🧠 Nasıl Çalışır?

1. **Belgeleri Parçalara Ayırma** (`ingest.py`)  
   - `.txt` ve `.pdf` dosyalarını okur, anlamlı bölümlere ayırır.  
   - Her parçanın **vektör temsilini** (embedding) yerel bir modelle oluşturur.  
   - Vektörleri ve metinleri **SQLite** veritabanında saklar.

2. **Sorgu ve Cevap Üretme** (`rag.py`)  
   - Sorunuzun vektörünü çıkarır.  
   - Veritabanındaki tüm parçalarla benzerlik hesaplar, en ilgili olanları seçer.  
   - Seçilen parçaları bir bağlam olarak yerel LLM’ye (büyük dil modeli) sunar.  
   - Model, yalnızca bu bağlama dayanarak cevabı oluşturur — tahmin yapmaz, uydurmaz.

3. **Kullanıcı Arayüzü** (`app.py`)  
   - **Streamlit** ile hazırlanmış basit bir sohbet arayüzü.  
   - Soruları yazın, cevabı ve hangi belge parçalarından yararlanıldığını görün.

---

## ✨ Özellikler

- ✅ **Tamamen çevrimdışı** — internet bağlantısı gerektirmez.  
- ✅ **Yerel modeller** — embedding ve dil modelleri bilgisayarınızda çalışır.  
- ✅ **SQLite tabanlı** — hafif, taşınabilir, kurulum gerektirmez.  
- ✅ **Kaynak gösterimi** — hangi belge ve sayfadan alıntı yapıldığını gösterir.  
- ✅ **Kolay özelleştirme** — konu adı, eşik değerleri, parça boyutları `.env` veya `config.py` ile ayarlanabilir.  
- ✅ **Streamlit arayüzü** — etkileşimli ve görsel.

---

## 🔧 Gereksinimler

- Python 3.10 veya üzeri  
- Yerel bir **OpenAI uyumlu API sunucusu** (ör. [LocalAI](https://github.com/mudler/LocalAI), [llama.cpp](https://github.com/ggerganov/llama.cpp) ile OpenAI uyumlu sunucu, veya [Foundry Local](https://learn.microsoft.com/en-us/azure/foundry-local/))  
  > Bu proje, OpenAI Python kütüphanesi üzerinden API çağrısı yapar. Herhangi bir OpenAI uyumlu sunucu ile çalışır.

- Bağımlılıklar: `streamlit`, `openai`, `pypdf`, `numpy`, `tqdm`

---

## 📦 Kurulum

1. **Depoyu klonlayın**
   ```bash
   git clone https://github.com/kullanici/yerel-rag-asistani.git
   cd yerel-rag-asistani
Sanal ortam oluşturun ve etkinleştirin (önerilir)

bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
Gerekli paketleri yükleyin

bash
pip install -r requirements.txt
Eğer requirements.txt yoksa, manuel olarak şunları kurun:
pip install streamlit openai pypdf numpy tqdm
Yerel API sunucusunu başlatın
Örneğin LocalAI kullanıyorsanız:

bash
local-ai run
Veya Foundry Local ile:

bash
foundry-local start
Sunucunuzun http://127.0.0.1:49575/v1 adresinde çalıştığından emin olun (varsayılan).
Veri klasörünü hazırlayın
data/documents/ dizini oluşturun ve içine .txt veya .pdf belgelerinizi koyun.
Belgeleri işleyin (embedding oluşturma)

bash
python ingest.py
Bu komut, tüm belgeleri parçalara ayıracak, embedding’lerini çıkaracak ve data/vector_store.db dosyasına kaydedecektir.
🚀 Çalıştırma

Streamlit Arayüzü (önerilir)

bash
streamlit run app.py
Tarayıcınızda http://localhost:8501 açılır. Sorularınızı yazıp cevap alabilir, ayrıca hangi belge parçalarının kullanıldığını yan menüden inceleyebilirsiniz.

Komut Satırı Sorgulama

bash
python rag.py
Bu betik, terminal üzerinden soru sormanızı sağlar.

Sadece Retrieval Testi

bash
python retrieve.py
Bu betik, verilen bir sorgu için en ilgili parçaları listeler (LLM çağrısı yapmaz).

⚙️ Yapılandırma

Tüm ayarlar config.py dosyasında veya çevre değişkenleriyle yapılabilir.

Değişken	Açıklama	Varsayılan
API_BASE_URL	Yerel OpenAI uyumlu sunucu adresi	http://127.0.0.1:49575/v1
API_KEY	API anahtarı (yerelde herhangi bir değer olabilir)	dummy
EMBEDDING_MODEL	Embedding model adı	qwen3-embedding-0.6b
LLM_MODEL	Dil modeli adı	phi-3.5-mini
SUBJECT_NAME	Arayüzde görünen konu adı	Anayasa
TOP_K	Cevap üretirken kullanılacak parça sayısı	8
DISPLAY_TOP_K	Kenar çubuğunda gösterilecek parça sayısı	8
RELEVANCE_THRESHOLD	Benzerlik eşiği (bu değerin altındaki parçalar kullanılmaz)	0.32
CHUNK_SIZE	Parça uzunluğu (karakter)	400
CHUNK_OVERLAP	Parçalar arası örtüşme	50
Çevre değişkenlerini .env dosyası ile de kullanabilirsiniz (örnek .env.example).

📁 Dosya Yapısı

text
.
├── app.py                # Streamlit arayüzü
├── config.py             # Tüm yapılandırma
├── ingest.py             # Belge işleme ve embedding oluşturma
├── rag.py                # Sorgu-cevap mantığı (retrieval + generation)
├── retrieve.py           # Sadece retrieval testi için
├── data/
│   ├── documents/        # Buraya .txt ve .pdf belgelerinizi koyun
│   └── vector_store.db   # SQLite veritabanı (otomatik oluşur)
└── README.md
❓ Sık Karşılaşılan Sorunlar

OpenAI API bağlantı hatası
Yerel API sunucunuzun çalıştığından ve API_BASE_URL adresinin doğru olduğundan emin olun.
Model bulunamadı
Yerel sunucunuza uygun model adlarını config.py'da doğru girin. Örneğin Foundry Local için phi-3.5-mini, LocalAI için gpt-3.5-turbo gibi.
Hiç parça bulunamadı
data/documents/ içinde desteklenen formatlarda dosya olduğundan ve ingest.py'yi başarıyla çalıştırdığınızdan emin olun.
Cevap hep "Bu bilgi notlarımda yok."
Ya soru belgelerde yoktur, ya da eşik değeri (RELEVANCE_THRESHOLD) çok yüksektir. Düşürmeyi deneyin.
🧪 Geliştirme ve Katkı

Projeyi geliştirmek veya kişisel ihtiyaçlarınıza uyarlamak için:

config.py'daki parametreleri değiştirin.
Kendi embedding veya dil modelinizi kullanmak için API adresini güncelleyin.
ingest.py'deki parçalama mantığını değiştirerek farklı boyutlar deneyin.
Streamlit arayüzünü özelleştirin (app.py).
Katkılarınızı memnuniyetle karşılarız — lütfen bir pull request açın veya issue bildirin.

📄 Lisans

Bu proje MIT Lisansı ile lisanslanmıştır. Aşağıdaki lisans metnini kabul ederek kullanabilir, kopyalayabilir, değiştirebilir ve dağıtabilirsiniz.

Proje kök dizinine LICENSE adında bir dosya oluşturup aşağıdaki metni kopyalamanız önerilir.
text
MIT License

Copyright (c) 2026 [Buraya kendi adınızı veya kurum adınızı yazın]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
🙏 Teşekkür

Bu proje, Microsoft AI Summer Innovators Internship Program kapsamında geliştirilmiştir.
Beni bu harika programa kabul eden, süreç boyunca destekleyen ve yönlendiren Barbaros Günay’a içten teşekkürlerimi sunarım. Programın sağladığı eğitim, kaynaklar ve ilham sayesinde bu projeyi hayata geçirme fırsatı buldum.

Herhangi bir sorunuz veya öneriniz varsa lütfen iletişime geçin.
İyi sorgulamalar! 🎓

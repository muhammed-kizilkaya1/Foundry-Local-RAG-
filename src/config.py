import os

# ==============================
# API Ayarları (yerel/offline model sunucusu)
# ==============================
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:49575/v1")
API_KEY = os.getenv("API_KEY", "dummy")

# ==============================
# Model Ayarları
# ==============================
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding-0.6b")
LLM_MODEL = os.getenv("LLM_MODEL", "phi-3.5-mini")

# ==============================
# Uygulama Ayarları
# ==============================
SUBJECT_NAME = os.getenv("SUBJECT_NAME", "Anayasa")          # Arayüzde gösterilen konu adı
APP_TITLE = f" {SUBJECT_NAME} Asistanı"

TOP_K = int(os.getenv("TOP_K", "8"))                          # Cevap üretirken kullanılacak parça sayısı
DISPLAY_TOP_K = int(os.getenv("DISPLAY_TOP_K", "8"))          # Kenar çubuğunda gösterilecek parça sayısı
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.4")) 

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# ==============================
# Veri Yolları
# ==============================
DATA_DIR = "../data"
DB_PATH = os.path.join(DATA_DIR, "vector_store.db")
DOC_FOLDER = os.path.join(DATA_DIR, "documents") 

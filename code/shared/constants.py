# ======================== IMPORTS ========================
import os
import sys
from enum import Enum
from pathlib import Path
from dotenv import load_dotenv
from shared.utils import load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))


# ======================= CONSTANTS =======================
# Configurations from config.yaml
config = load_config()
ES_HOST                : str  = config.get("elasticsearch", {}).get("host", "localhost")
ES_PORT                : int  = config.get("elasticsearch", {}).get("port", 9200)
ES_SCHEME              : str  = config.get("elasticsearch", {}).get("scheme", "http")
REDIS_HOST             : str  = config.get("redis", {}).get("host", "localhost")
REDIS_PORT             : int  = config.get("redis", {}).get("port", 6379)
REDIS_DB               : int  = config.get("redis", {}).get("db", 0)

DATA_SETTINGS          : dict = config.get("data", {})
PREPROCESSING_SETTINGS : dict = config.get("preprocessing", {})
CUST_INDEX_SETTINGS    : dict = config.get("cust_index_settings", {})

MAX_RESULTS            : int  = config.get("max_results", 50)
SEARCH_FIELDS          : str  = config.get("search_fields", ["text"])
MAX_NUM_DOCUMENTS      : int  = config.get("max_num_documents", -1)
TOP_K_WORDS_THRESHOLD  : int  = config.get("top_k_words_threshold", 50)
CHUNK_SIZE             : int  = config.get("elasticsearch", {}).get("chunk_size", 500)
RANKING_SCORE_THRESHOLD: float = config.get("ranking_score_threshold", 0.1)

STORAGE_DIR            : str  = PROJECT_ROOT / config.get("storage_dir", "storage")
TEMP_DIR               : str  = PROJECT_ROOT / config.get("temp_dir", "temp")
OUTPUT_DIR             : str  = PROJECT_ROOT / config.get("output_dir", "output")
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Environment variables
load_dotenv()
USERNAME: str = os.getenv("USERNAME")
PASSWORD: str = os.getenv("PASSWORD")
API_KEY : str = os.getenv("API_KEY")


# ========================= ENUMS =========================
class IndexInfo(Enum):
    """
    Enumeration for different index information types.
    """
    NONE     : int = 0 # Just a placeholder
    BOOLEAN  : int = 1
    WORDCOUNT: int = 2
    TFIDF    : int = 3


class DataStore(Enum):
    """
    Enumeration for different data storage types.
    """
    NONE   : int = 0 # Just a placeholder
    CUSTOM : int = 1
    ROCKSDB: int = 2
    REDIS  : int = 3


class Compression(Enum):
    """
    Enumeration for different compression types."""
    NONE: int = 0
    CODE: int = 1
    CLIB: int = 2


class Optimizations(Enum):
    """
    Enumeration for different optimization techniques."""
    NONE     : str = '0'
    OPTIMISED: str = 'O'


class QueryProc(Enum):
    """
    Enumeration for different query processing techniques.
    """
    NONE: str = '0' # Just a placeholder
    TERM: str = 'T'
    DOC : str = 'D'


class StatusCode(Enum):
    """
    Enumeration for different status codes used in the indexing and retrieval system.
    """
    SUCCESS              : int = 0
    CONNECTION_FAILED    : int = 1000
    ERROR_ACCESSING_INDEX: int = 1001
    INVALID_INPUT        : int = 1002
    INDEXING_FAILED      : int = 1003
    FAILED_TO_REMOVE_FILE: int = 1004
    QUERY_FAILED         : int = 1005
    INDEX_NOT_FOUND      : int = 1006
    INDEX_ALREADY_EXISTS : int = 1007
    INDEX_NOT_LOADED     : int = 1008
    
    UNKNOWN_ERROR        : int = 9999


class IndexType(Enum):
    """
    Enumeration for different index types.
    """
    ESIndex    : int = 1
    CustomIndex: int = 2


class DatasetType(Enum):
    """
    Enumeration for different dataset types.
    """
    News     : int = 1
    Wikipedia: int = 2

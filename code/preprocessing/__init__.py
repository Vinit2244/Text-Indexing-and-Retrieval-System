import ssl
import nltk
from .preprocessor import Preprocessor


# --- Fix SSL verification issue while downloading stopwords and wordnet ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


# Download required NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

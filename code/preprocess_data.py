# ======================== IMPORTS ========================
from typing import List
from shared.utils import load_config, Style
from dataset_managers import NewsDataset, WikipediaDataset
from shared.constants import MAX_NUM_DOCUMENTS, PREPROCESSING_SETTINGS, PROJECT_ROOT


# ======================= FUNCTIONS =======================
def preprocess_data(config: dict) -> None:
    """
    About:
    ------
        Preprocesses the datasets (News and Wikipedia) based on the settings provided in the configuration.

    Args:
    -----
        config (dict): Configuration dictionary containing preprocessing settings.

    Returns:
    --------
        None
    """

    # Lower
    lowercase         : bool             = PREPROCESSING_SETTINGS["lowercase"]
    # Remove
    stopword_langs    : List[str] | None = PREPROCESSING_SETTINGS["stopwords"]["languages"]
    rem_stop          : bool             = True if stopword_langs else False
    rem_punc          : bool             = PREPROCESSING_SETTINGS["remove_punctuation"]
    rem_num           : bool             = PREPROCESSING_SETTINGS["remove_numbers"]
    rem_special       : bool             = PREPROCESSING_SETTINGS["remove_special_characters"]
    # Stemming
    stemming_algo     : str | None       = PREPROCESSING_SETTINGS["stemming"]["algorithm"]
    stem              : bool             = True if stemming_algo else False
    # Lemmatization
    lemmatization_algo: str | None       = PREPROCESSING_SETTINGS["lemmatization"]["algorithm"]
    lemmatize         : bool             = True if lemmatization_algo else False

    max_num_documents: int =MAX_NUM_DOCUMENTS if MAX_NUM_DOCUMENTS is not None else -1
    print(f"{Style.FG_YELLOW}Using Settings for preprocessing: \n\tLowercase: {lowercase}, \n\tRemove Stopwords: {rem_stop} ({stopword_langs}), \n\tRemove Punctuation: {rem_punc}, \n\tRemove Numbers: {rem_num}, \n\tRemove Special Characters: {rem_special}, \n\tStemming: {stem} ({stemming_algo}), \n\tLemmatization: {lemmatize} ({lemmatization_algo}), \n\tMax documents: {max_num_documents}{Style.RESET}. \nTo change, modify config.yaml file.\n")

    print(f"{Style.FG_CYAN}Preprocessing news dataset...{Style.RESET}")
    path_to_news_dataset: str = PROJECT_ROOT / config["data"]["news"]["path"]
    unzip: bool = config["data"]["news"]["unzip"]
    NewsDataset(path_to_news_dataset, max_num_documents, unzip).preprocess(lowercase, rem_stop, stopword_langs, rem_punc, rem_num, rem_special, stem, stemming_algo, lemmatize, lemmatization_algo)
    print(f"{Style.FG_GREEN}Preprocessing of news dataset completed.\n{Style.RESET}")

    print(f"{Style.FG_CYAN}Preprocessing wikipedia dataset...{Style.RESET}")
    path_to_wikipedia_dataset: str = PROJECT_ROOT / config["data"]["wikipedia"]["path"]
    WikipediaDataset(path_to_wikipedia_dataset, max_num_documents).preprocess(lowercase, rem_stop, stopword_langs, rem_punc, rem_num, rem_special, stem, stemming_algo, lemmatize, lemmatization_algo)
    print(f"{Style.FG_GREEN}Preprocessing of wikipedia dataset completed.\n{Style.RESET}")


# ========================= MAIN ==========================
def main():
    config = load_config()
    preprocess_data(config)


if __name__ == "__main__":
    main()

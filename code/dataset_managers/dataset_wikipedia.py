# ======================== IMPORTS ========================
import os
import pandas as pd
from tqdm import tqdm
from typing import List
from .dataset_base import Dataset
from collections import defaultdict
from shared.constants import PROJECT_ROOT
from shared.utils import Style, load_config, get_word_freq_dist


# ======================== CLASSES ========================
class WikipediaDataset(Dataset):
    """
    Wikipedia Dataset Handler.
    Manages downloading, word frequency calculation, and preprocessing of the Wikipedia dataset stored in Parquet
    """

    def __init__(self, data_path: str, max_num_docs: int) -> None:
        self.data_path = data_path
        self.max_num_docs = max_num_docs

        os.makedirs(self.data_path, exist_ok=True)
        self.wikipedia_parquet_files: List[str] = [f for f in os.listdir(self.data_path) if f.endswith('.parquet')]
        
        # Fet total number of rows across all files (to use in tqdm progress bar)
        self.total_rows = sum(pd.read_parquet(os.path.join(self.data_path, f), engine='pyarrow').shape[0] for f in self.wikipedia_parquet_files)

        if self.max_num_docs != -1:
            self.total_rows = min(self.total_rows, self.max_num_docs)

    # Public functions
    def download_dataset(self) -> None:
        """
        About:
        ------
            Downloads the Wikipedia dataset to the specified path.
            NOTE: Due to the large size of the dataset, automatic downloading is not implemented.

        Args:
        -----
            None

        Returns:
        --------
            None
        """

        os.makedirs(self.data_path, exist_ok=True)
        print(f"{Style.FG_RED + Style.BG_YELLOW + Style.BOLD}Manually download .parquet files of wikipedia dataset from https://huggingface.co/datasets/wikimedia/wikipedia/tree/main and save the files at {self.data_path}{Style.RESET}")

    def get_attributes(self) -> List[str]:
        """
        About:
        ------
            Retrieves the list of attributes (columns) present in the Parquet files of the dataset.

        Args:
        -----
            None

        Returns:
        --------
            A list of attribute names (columns) present in the dataset.
        """
        
        # Get attributes from the first parquet file in the dataset
        if not self.wikipedia_parquet_files:
            return []
        
        first_file_path: str = os.path.join(self.data_path, self.wikipedia_parquet_files[0])
        df = pd.read_parquet(first_file_path, engine='pyarrow')
        return list(df.columns)

    def calculate_word_frequency(self) -> dict:
        """
        About:
        ------
            Calculates the overall word frequency distribution across all Parquet files in the dataset.

        Args:
        -----
            None

        Returns:
        --------
            A dictionary with words as keys and their corresponding frequencies as values.
        """
        
        freq: dict = defaultdict(int)

        curr_row_count = 0
        with tqdm(total=self.total_rows, desc="Calculating word frequencies") as pbar:
            for parquet_file in self.wikipedia_parquet_files:
                break_flag = False
                parquet_file_path: str = os.path.join(self.data_path, parquet_file)
                df = pd.read_parquet(parquet_file_path, engine='pyarrow')
                for text in df["text"]:
                    if curr_row_count == self.max_num_docs:
                        break_flag = True
                        break
                    item_freq = get_word_freq_dist(text)
                    for word, count in item_freq.items():
                        freq[word] += count
                    pbar.update(1)  # update progress bar for each row
                    curr_row_count += 1
                if break_flag:
                    break

        return freq

    def preprocess(self, lowercase: bool, rem_stop: bool, stopword_langs: List[str], rem_punc: bool, rem_num: bool, rem_special: bool, stem: bool, stemming_algo: str, lemmatize: bool, lemmatization_algo: str) -> None:
        """
        About:
        ------
            Preprocesses the text data in each Parquet file according to the specified parameters.
            Modifies the "text" field in each Parquet file in place.

        Args:
        -----
            lowercase (bool): Whether to convert text to lowercase.
            rem_stop (bool): Whether to remove stopwords.
            stopword_langs (List[str]): List of languages for stopword removal. Can include "auto".
            rem_punc (bool): Whether to remove punctuation.
            rem_num (bool): Whether to remove numbers.
            rem_special (bool): Whether to remove special characters.
            stem (bool): Whether to apply stemming.
            stemming_algo (str): The stemming algorithm to use.
            lemmatize (bool): Whether to apply lemmatization.
            lemmatization_algo (str): The lemmatization algorithm to use.

        Returns:
        --------
            A dictionary with words as keys and their corresponding frequencies as values.
        """
        
        from preprocessing import Preprocessor
        
        freq: dict = defaultdict(int)

        curr_row_count = 0
        preprocessor = Preprocessor()
        with tqdm(total=self.total_rows, desc="Preprocessing text") as pbar:
            for parquet_file in self.wikipedia_parquet_files:
                break_flag = False
                parquet_file_path: str = os.path.join(self.data_path, parquet_file)

                df = pd.read_parquet(parquet_file_path, engine='pyarrow')
                processed_texts = []
                for text in df["text"]:
                    if curr_row_count == self.max_num_docs:
                        # Append remaining unprocessed texts as is and break out
                        break_flag = True
                        remaining_texts = df["text"].iloc[len(processed_texts):].tolist()
                        processed_texts.extend(remaining_texts)
                        break
                    if lowercase:
                        text = preprocessor.lowercase(text)
                    if rem_stop:
                        for lang in stopword_langs:
                            if lang.strip().lower() == "auto":
                                continue
                            text = preprocessor.remove_stopwords(text, lang)
                    if rem_punc or rem_num or rem_special:
                        text = preprocessor.remove(text, rem_punc, rem_num, rem_special)
                    if stem:
                        text = preprocessor.stem(text, stemming_algo)
                    if lemmatize:
                        text = preprocessor.lemmatize(text, lemmatization_algo)

                    processed_texts.append(text)
                    pbar.update(1)
                    curr_row_count += 1

                # Update dataframe with processed text
                df["text"] = processed_texts

                # Overwrite same parquet file (or change filename if you want to keep original)
                df.to_parquet(parquet_file_path, index=False, engine='pyarrow')

                if break_flag:
                    break
        return freq

    def get_files(self, attributes: List[str]) -> List[tuple[str, dict]]:
        """
        About:
        ------
            Generator that yields tuples of (unique identifier, payload dictionary) for each Parquet file in the dataset.
            The unique identifier is taken from the first attribute in the provided list, and the payload dictionary contains the remaining attributes.
        
        Args:
        -----
            attributes (List[str]): List of attribute names to extract from each Parquet file. The first attribute is treated as the unique identifier.
        
        Yields:
        --------
            Yields tuples of (unique identifier, payload dictionary) for each Parquet file in the dataset.
        """
        
        files: List[tuple[str, dict]] = []

        seen_ids = set()

        curr_row_count = 0
        for parquet_file in self.wikipedia_parquet_files:
            break_flag = False
            parquet_file_path: str = os.path.join(self.data_path, parquet_file)
            df = pd.read_parquet(parquet_file_path, engine='pyarrow')
            for _, row in df.iterrows():
                if curr_row_count == self.max_num_docs:
                    break_flag = True
                    break
                
                file_id: str = str(row[attributes[0]])  # First attribute is unique id
                if file_id in seen_ids:
                    curr_row_count += 1
                    continue  # Skip duplicate ids
                seen_ids.add(file_id)
                content: dict = {attr: row[attr] for attr in attributes[1:]}  # Rest are content attributes
                
                files.append((file_id, content))
                curr_row_count += 1
            
            if break_flag:
                break
        
        return files


# ================== HELPER FUNCTIONS ====================
def get_wikipedia_dataset_handler(max_num_docs: int, verbose: bool=True) -> WikipediaDataset:
    """
    About:
    ------
        Helper function to create and return a WikipediaDataset handler using configuration settings.

    Args:
    -----
        max_num_docs (int): Maximum number of documents to handle. Use -1 for all documents.
        verbose (bool): Whether to print status messages.

    Returns:
    --------
        WikipediaDataset: An instance of the WikipediaDataset handler.
    """
    
    config = load_config()
    
    data_path: str = PROJECT_ROOT / config["data"]["wikipedia"]["path"]
    
    if verbose:
        print(f"{Style.FG_YELLOW}Using \n\tMax docs: {max_num_docs}.{Style.RESET}\nTo change, modify config.yaml file.\n")
    
    return WikipediaDataset(data_path, max_num_docs)

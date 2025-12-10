# ======================== IMPORTS ========================
import os
import json
import zipfile
import subprocess
from tqdm import tqdm
from .dataset_base import Dataset
from collections import defaultdict
from typing import List, Tuple, Generator
from shared.constants import PROJECT_ROOT
from shared.utils import Style, load_config, get_word_freq_dist


# ======================== CLASSES ========================
class NewsDataset(Dataset):
    """
    Handler for the News dataset from Webhose.
    Provides methods to download the dataset, calculate word frequencies, preprocess text data, and iterate over files.
    """

    def __init__(self, data_path: str, max_num_docs: int, unzipped: bool) -> None:
        self.data_path = data_path
        self.max_num_docs = max_num_docs
        self.unzipped = unzipped
        pass

    # Private functions (First returns the total length of the iterator, second is the actual iterator)
    def _file_iterator(self) -> Generator:
        """
        About:
        ------
            Generator that yields file objects for each JSON file in the dataset.
            Handles both unzipped and zipped datasets.
            Yields the total number of files as the first value for progress tracking.

        Args:
        -----
            None    
        
        Yields:
        -------
            file object: Open file object for each JSON file in the dataset.
        """

        if self.unzipped:
            all_folders: List[str] = os.listdir(self.data_path)

            # Flatten list of all JSON files across all folders
            all_json_files_paths: List[str] = []
            for folder in all_folders:
                folder_path = os.path.join(self.data_path, folder)
                # Make sure it's a directory
                if os.path.isdir(folder_path):
                    json_files = os.listdir(folder_path)
                    for json_file in json_files:
                        all_json_files_paths.append(os.path.join(folder_path, json_file))
            
            # Fet total number of files (to use in tqdm progress bar)
            if self.max_num_docs != -1:
                all_json_files_paths = all_json_files_paths[:self.max_num_docs]
            total_files = len(all_json_files_paths)
            yield total_files

            # Iterate over all files with tqdm
            for json_file_path in all_json_files_paths:
                with open(json_file_path, 'r') as f:
                    yield f
        
        else:
            zipped_folders_path: str = os.path.join(self.data_path, "News_Datasets")
            all_zipped_folders: List[str] = os.listdir(zipped_folders_path)

            # Flatten all JSON files inside all zip folders
            all_json_entries = []

            for zipped_folder in all_zipped_folders:
                if zipped_folder.endswith(".zip"):
                    zip_path = os.path.join(zipped_folders_path, zipped_folder)
                    with zipfile.ZipFile(zip_path, "r") as z:
                        # store both zip path and internal JSON file name
                        for json_file in z.namelist():
                            all_json_entries.append((zip_path, json_file))

            # Fet total number of files (to use in tqdm progress bar)
            if self.max_num_docs != -1:
                all_json_entries = all_json_entries[:self.max_num_docs]
            total_files = len(all_json_entries)
            yield total_files

            # Iterate over all JSON files with tqdm
            for zip_path, json_file in all_json_entries:
                with zipfile.ZipFile(zip_path, "r") as z:
                    with z.open(json_file) as f:
                        yield f

    def _unzip_and_clean_dataset(self) -> None:
        """
        About:
        ------
            Unzips all zipped folders in the "News_Datasets" directory and cleans up unnecessary files.
            Removes all files except the "News_Datasets" folder before unzipping.

        Args:
        -----
            None

        Returns:
        -------
            None
        """

        print(f"{Style.FG_CYAN}Unzipping and cleaning news dataset...{Style.RESET}")

        # Remove all the files except the "News_Datasets" folder
        all_items: list = os.listdir(self.data_path)
        for item in all_items:
            if item != "News_Datasets":
                item_path: str = os.path.join(self.data_path, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    subprocess.run(["rm", "-rf", item_path], check=True)
        
        # Unzip all the zipped folders in "News_Datasets" and remove the zipped files
        path_to_zipped_folders: str = os.path.join(self.data_path, "News_Datasets")
        all_zipped_folders: list = os.listdir(path_to_zipped_folders)
        unzipped_count = 0
        for folder in all_zipped_folders:
            folder_path: str = os.path.join(path_to_zipped_folders, folder)
            if folder_path.endswith(".zip"):
                try:
                    subprocess.run(["unzip", "-o", folder_path, "-d", self.data_path], check=True)
                    unzipped_count += 1
                    os.remove(folder_path)
                except:
                    subprocess.run(["rm", "-rf", folder_path], check=True)
        print(f"{Style.FG_GREEN}Unzipped {unzipped_count}/{len(all_zipped_folders)}{Style.RESET}")

        print(f"{Style.FG_GREEN}News dataset unzipped and cleaned at {self.data_path}\n{Style.RESET}")

    # Public functions
    def download_dataset(self) -> None:
        """
        About:
        ------
            Downloads the News dataset from the Webhose GitHub repository.
            If unzipped is True, it also unzips and cleans the dataset.

        Args:
        -----
            None

        Returns:
        -------
            None
        """

        print(f"{Style.FG_CYAN}Downloading news dataset...{Style.RESET}")

        # Make sure the destination directory exists
        os.makedirs(self.data_path, exist_ok=True)

        # Clone the repository
        repo_url = "https://github.com/Webhose/free-news-datasets.git"
        subprocess.run(["git", "clone", repo_url, self.data_path], check=True)

        if self.unzipped:
            print(f"{Style.FG_CYAN}Unzipping and cleaning news dataset...{Style.RESET}")
            self._unzip_and_clean_dataset()

        print(f"{Style.FG_GREEN}News dataset downloaded at {self.data_path}\n{Style.RESET}")

    def get_attributes(self) -> List[str]:
        """
        About:
        ------
            Retrieves the list of attributes (keys) present in the JSON files of the dataset.
        
        Args:
        -----
            None

        Returns:
        -------
            List[str]: List of attribute names found in the JSON files.
        """

        # Get attributes from the first JSON file in the dataset
        for f in self._file_iterator():
            # Ignoring the first value which is the total count
            if type(f) is int:
                continue
            data = json.load(f)
            return list(data.keys())
        return []

    def calculate_word_frequency(self) -> dict:
        """
        About:
        ------
            Calculates the overall word frequency distribution across all JSON files in the dataset.

        Args:
        -----
            None

        Returns:
        -------
            dict: A dictionary with words as keys and their corresponding frequencies as values.
        """

        freq: dict = defaultdict(int)
        
        # Updates the overall frequency dictionary with the frequency from a single json file
        def update_freq_dict(f):
            """
            About:
            ------
                Updates the overall frequency dictionary with the frequency from a single json file.

            Args:
            -----
                f: File object of the JSON file.

            Returns:
            -------
                None
            """

            text: str = json.load(f)["text"]
            file_freq: dict = get_word_freq_dist(text)
            for word, count in file_freq.items():
                freq[word] += count
        
        # Fet total number of files (to use in tqdm progress bar)
        total_files = next(self._file_iterator())

        for f in tqdm(self._file_iterator(), total=total_files, desc="Calculating word frequencies"):
            # Ignoring the first value which is the total count
            if type(f) is int:
                continue
            update_freq_dict(f)

        return freq

    def preprocess(self, lowercase: bool, rem_stop: bool, stopword_langs: List[str], rem_punc: bool, rem_num: bool, rem_special: bool, stem: bool, stemming_algo: str, lemmatize: bool, lemmatization_algo: str) -> None:
        """
        About:
        ------
            Preprocesses the text data in each JSON file according to the specified parameters.
            Modifies the "text" field in each JSON file in place.

        Args:
        -----
            lowercase (bool): Whether to convert text to lowercase.
            rem_stop (bool): Whether to remove stopwords.
            stopword_langs (List[str]): List of languages for stopword removal. Can include "auto".
            rem_punc (bool): Whether to remove punctuation.
            rem_num (bool): Whether to remove numbers.
            rem_special (bool): Whether to remove special characters.
            stem (bool): Whether to apply stemming.
            stemming_algo (str): Stemming algorithm to use (e.g., "porter", "snowball").
            lemmatize (bool): Whether to apply lemmatization.
            lemmatization_algo (str): Lemmatization algorithm to use (e.g., "wordnet").

        Returns:
        -------
            None
        """

        from preprocessing import Preprocessor
        
        preprocessor = Preprocessor()
        total_files = next(self._file_iterator())

        for f in tqdm(self._file_iterator(), total=total_files, desc="Preprocessing files"):
            # Ignoring the first value which is the total count
            if type(f) is int:
                continue

            data = json.load(f)
            text: str = data["text"]
            if lowercase:
                text = preprocessor.lowercase(text)
            if rem_stop:
                if "auto" in stopword_langs:
                    try:
                        lang: str = data["language"]
                        text = preprocessor.remove_stopwords(text, lang)
                    except:
                        # If language tag not found, skip "auto" and use other specified languages
                        for lang in stopword_langs:
                            if lang.strip().lower() == "auto":
                                continue
                            text = preprocessor.remove_stopwords(text, lang)
                else:
                    # Use all specified languages (no "auto")
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
            data["text"] = text

            # Write back to file (overwriting original content)
            with open(f.name, 'w') as out_f:
                json.dump(data, out_f)

    def get_files(self, attributes: List[str]) -> Generator[Tuple[str, dict], None, None]:
        """
        About:
        ------
            Generator that yields tuples of (unique identifier, payload dictionary) for each JSON file in the dataset.
            The unique identifier is taken from the first attribute in the provided list, and the payload dictionary contains the remaining attributes.
        
        Args:
        -----
            attributes (List[str]): List of attribute names to extract from each JSON file. The first attribute is treated as the unique identifier.
        
        Yields:
        -------
            Tuple[str, dict]: A tuple containing the unique identifier (str) and a dictionary of the remaining attributes.
        """

        seen_ids = set()

        for f in self._file_iterator():
            # Ignoring the first value which is the total count
            if type(f) is int:
                continue
            data = json.load(f)

            try:
                # Assuming first attribute is always unique identifier
                uuid = str(data[attributes[0]])
                if uuid in seen_ids:
                    continue  # Skip duplicate ids
                seen_ids.add(uuid)
                payload = {attr: data[attr] for attr in attributes[1:]}
                
                # Returning a dict so that we can easily extend it in future if needed
                file_info = (uuid, payload)
            except KeyError as e:
                print(f"{Style.FG_RED}Warning: Attribute {e} not found in file {f.name}. Skipping this file.{Style.RESET}")
                continue
            
            yield file_info


# ================== HELPER FUNCTIONS ====================
def get_news_dataset_handler(max_num_docs: int, verbose: bool=True) -> NewsDataset:
    """
    About:
    ------
        Helper function to create and return a NewsDataset handler using configuration settings.

    Args:
    -----
        max_num_docs (int): Maximum number of documents to handle. Use -1 for all documents.
        verbose (bool): Whether to print status messages.

    Returns:
    -------
        NewsDataset: An instance of the NewsDataset handler.
    """
    
    config: dict = load_config()

    path: str = PROJECT_ROOT / config["data"]["news"]["path"]
    unzipped: bool = config["data"]["news"]["unzip"]

    if verbose:
        print(f"{Style.FG_YELLOW}Using \n\tMax docs: {max_num_docs}, \n\tUnzipped: {unzipped}{Style.RESET}. \nTo change, modify config.yaml file.\n")

    return NewsDataset(path, max_num_docs, unzipped)

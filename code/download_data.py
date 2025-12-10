# ======================== IMPORTS ========================
from shared.utils import Style
from shared.constants import DATA_SETTINGS, PROJECT_ROOT
from dataset_managers import NewsDataset, WikipediaDataset


# ========================= MAIN ==========================
def main() -> None:
    """
    About:
    -----
        Main function to download datasets as specified in the configuration.yaml file.
        
    Args:
    -----
        None

    Returns:
    --------
        None
    """
    
    # Download and unzip news dataset
    path_to_news_dataset: str = PROJECT_ROOT / DATA_SETTINGS["news"]["path"]
    unzip: bool = DATA_SETTINGS["news"]["unzip"]
    print(f"{Style.FG_YELLOW}Using \n\tPath to News: {path_to_news_dataset}{Style.RESET}, \n\tUnzip: {unzip}. \nTo change, modify config.yaml file.\n")

    news_dataset_handler = NewsDataset(path_to_news_dataset, -1, unzip)
    news_dataset_handler.download_dataset()

    # Download wikipedia dataset
    path_to_wikipedia_dataset: str = PROJECT_ROOT / DATA_SETTINGS["wikipedia"]["path"]
    print(f"{Style.FG_YELLOW}Using \n\tPath to Wiki: {path_to_wikipedia_dataset}{Style.RESET}. \nTo change, modify config.yaml file.\n")

    wikipedia_dataset_handler = WikipediaDataset(path_to_wikipedia_dataset, -1)
    wikipedia_dataset_handler.download_dataset()


if __name__ == "__main__":
    main()

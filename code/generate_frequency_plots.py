# ======================== IMPORTS ========================
import os
import argparse
from shared.utils import Style
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from dataset_managers import NewsDataset, WikipediaDataset
from shared.constants import DATA_SETTINGS, MAX_NUM_DOCUMENTS, TOP_K_WORDS_THRESHOLD, OUTPUT_DIR, PROJECT_ROOT


# ======================= FUNCTIONS =======================
def plot_frequency_distribution(freq_dict: Dict[str, int], k: int, title: str, xlabel: str, ylabel: str, output_file_path: str) -> None:
    """
    About:
    ------
        Plots the frequency distribution of words and saves the plot as an image file.
    
    Args:
    -----
        freq_dict (Dict[str, int]): Dictionary with words as keys and their frequencies as values.
        k (int): Number of top words to plot.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        output_file_path (str): Path to save the output plot image.
    
    Returns:
    --------
        None
    """
    
    def get_x_y(freq_dist: dict) -> Tuple[List[str], List[int]]:
        freqs: List[Tuple[str, int]] = [(word, count) for word, count in freq_dist.items()]
        freqs.sort(key=lambda x: x[1], reverse=True)
        x: List[str] = list()
        y: List[int] = list()
        for word, count in freqs:
            x.append(word)
            y.append(count)
        return x, y
    
    x, y = get_x_y(freq_dict)

    plt.figure(figsize=(10, 6))
    plt.bar(x[:k], y[:k])  # Plot only the top k for better visibility
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_file_path)
    plt.close()


# ========================= MAIN =========================
def main(args) -> None:
    """
    About:
    -----
        Main function to generate and save frequency distribution plots for news and wikipedia datasets.

    Args:
    -----
        args: Command line arguments containing data state information.

    Returns:
    --------
        None
    """

    print(f"{Style.FG_YELLOW}Using \n\tTop K words threshold: {TOP_K_WORDS_THRESHOLD}, \n\tMax documents: {MAX_NUM_DOCUMENTS}, \n\tOutput folder path: {OUTPUT_DIR}{Style.RESET}. \nTo change, modify config.yaml file.\n")
    os.makedirs(OUTPUT_DIR, exist_ok=True)


    # News Dataset
    print(f"{Style.FG_CYAN}Calculating word frequency for news dataset...{Style.RESET}")
    news_data_path: str = PROJECT_ROOT / DATA_SETTINGS["news"]["path"]
    unzipped: bool = DATA_SETTINGS["news"]["unzip"]
    print(f"{Style.FG_YELLOW}Using \n\tMax docs: {MAX_NUM_DOCUMENTS}, \n\tUnzipped: {unzipped}{Style.RESET}. \nTo change, modify config.yaml file.\n")

    news_dataset_handler = NewsDataset(news_data_path, MAX_NUM_DOCUMENTS, unzipped)
    news_data_freq_dist: dict = news_dataset_handler.calculate_word_frequency()
    plot_frequency_distribution(news_data_freq_dist,
                                TOP_K_WORDS_THRESHOLD,
                                "Word Frequency Distribution for News Dataset",
                                "Words",
                                "Frequencies",
                                os.path.join(OUTPUT_DIR, f"news_word_frequency_{args.data_state}_top_{TOP_K_WORDS_THRESHOLD}.png"))
    print(f"{Style.FG_GREEN}Frequency plot for news dataset saved at {OUTPUT_DIR}\n{Style.RESET}")


    # Wikipedia Dataset
    print(f"{Style.FG_CYAN}Calculating word frequency for wikipedia dataset...{Style.RESET}")
    wikipedia_data_path: str = PROJECT_ROOT / DATA_SETTINGS["wikipedia"]["path"]
    print(f"{Style.FG_YELLOW}Using Max docs: {MAX_NUM_DOCUMENTS}{Style.RESET}.\nTo change, modify config.yaml file.\n")

    wikipedia_dataset_handler = WikipediaDataset(wikipedia_data_path, MAX_NUM_DOCUMENTS)
    wikipedia_data_freq_dist: dict = wikipedia_dataset_handler.calculate_word_frequency()
    plot_frequency_distribution(wikipedia_data_freq_dist,
                                TOP_K_WORDS_THRESHOLD,
                                "Word Frequency Distribution for Wikipedia Dataset",
                                "Words",
                                "Frequencies",
                                os.path.join(OUTPUT_DIR, f"wikipedia_word_frequency_{args.data_state}_top_{TOP_K_WORDS_THRESHOLD}.png"))
    print(f"{Style.FG_GREEN}Frequency plot for wikipedia dataset saved at {OUTPUT_DIR}\n{Style.RESET}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_state", type=str, help="State of the data - preprocessed / raw")
    args = argparser.parse_args()
    
    main(args)

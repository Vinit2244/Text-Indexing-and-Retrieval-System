# ======================== IMPORTS ========================
import json
import random
import argparse
from shared.utils import Style
from typing import List, Dict
from shared.constants import MAX_NUM_DOCUMENTS, PROJECT_ROOT
from dataset_managers.dataset_news import get_news_dataset_handler
from dataset_managers.dataset_wikipedia import get_wikipedia_dataset_handler


# ======================== GLOBALS ========================
# Word pools based on frequency rank (to filter out stop-words and typos)
MIN_WORD_FREQ        = 10     # Ignore words with frequency less than this
HIGH_FREQ_RANK_START = 100    # Start high-freq pool after top 100 words (likely stop words)
HIGH_FREQ_RANK_END   = 500
MID_FREQ_RANK_START  = 501
MID_FREQ_RANK_END    = 2000
LOW_FREQ_RANK_START  = 2001
LOW_FREQ_RANK_END    = 10000

# Templates for generating diverse queries
# _H_ = High-freq word, _M_ = Mid-freq word, _L_ = Low-freq word
QUERY_TEMPLATES = [
    "_M_",
    "_L_",
    "_M_ AND _M_",
    "_L_ AND _L_",
    "_M_ OR _L_",
    "_H_ AND _M_",
    "_M_ AND NOT _H_",
    "_L_ AND NOT _M_",
    "(_M_ AND _M_) OR _L_",
    "(_H_ AND _M_) OR (_M_ AND _L_)",
    "(_M_ OR _L_) AND NOT _H_",
    "(_M_ AND _L_) AND (_M_ OR _L_)",
    "(_M_ AND NOT _H_) OR (_L_ AND NOT _M_)",
    "(_H_ OR _M_) AND (_L_ OR _M_)",
    "_L_ AND NOT _H_"
]


# ======================= FUNCTIONS =======================
def get_dataset_handler(dataset_name: str) -> object:
    """
    About:
    ------
        Retrieves the appropriate dataset handler based on the configuration settings.

    Args:
    -----
        None

    Returns:
    --------
        An instance of the dataset handler (NewsDataset or WikipediaDataset).
    """

    print(f"{Style.FG_CYAN}Loading dataset handler for: {dataset_name}{Style.RESET}")

    if dataset_name == "News":
        return get_news_dataset_handler(MAX_NUM_DOCUMENTS, verbose=False)
    elif dataset_name == "Wikipedia":
        return get_wikipedia_dataset_handler(MAX_NUM_DOCUMENTS, verbose=False)
    else:
        print(f"{Style.FG_RED}Error: Unknown dataset '{dataset_name}' in config.yaml{Style.RESET}")
        exit(1)


def get_word_pools(freq_dict: Dict[str, int]) -> Dict[str, List[str]]:
    """
    About:
    ------
        Creates word pools (high, mid, low frequency) based on the provided frequency dictionary.

    Args:
    -----
        freq_dict (Dict[str, int]): A dictionary mapping words to their frequencies.

    Returns:
    --------
        A dictionary with keys 'high', 'mid', 'low' mapping to lists of words in each frequency pool.
    """

    print("Sorting words by frequency...")
    # Sort by frequency (high to low) and filter out very rare words
    sorted_words = [
        word for word, freq in sorted(freq_dict.items(), key=lambda item: item[1], reverse=True)
        if freq >= MIN_WORD_FREQ
    ]

    # Create pools based on rank
    high_pool = sorted_words[HIGH_FREQ_RANK_START : HIGH_FREQ_RANK_END]
    mid_pool  = sorted_words[MID_FREQ_RANK_START : MID_FREQ_RANK_END]
    low_pool  = sorted_words[LOW_FREQ_RANK_START : LOW_FREQ_RANK_END]

    # In case the dataset is small and pools are empty, use fallbacks
    if not low_pool:
        low_pool = mid_pool
    if not mid_pool:
        mid_pool = high_pool
    if not high_pool:
        high_pool = mid_pool
    
    # If all are still empty, the dataset is too small or filtered
    if not high_pool and not mid_pool and not low_pool:
        print(f"{Style.FG_RED}Error: Not enough words found with minimum frequency ({MIN_WORD_FREQ}) to create word pools.{Style.RESET}")
        print("Tip: Try lowering MIN_WORD_FREQ or using a larger dataset.")
        exit(1)
    
    # Ensure all pools have at least one word to avoid random.choice errors
    if not low_pool: low_pool = mid_pool
    if not mid_pool: mid_pool = low_pool
    if not high_pool: high_pool = mid_pool

    print(f"Word pools created:")
    print(f"  {Style.FG_YELLOW}High-freq pool size: {len(high_pool)}{Style.RESET}")
    print(f"  {Style.FG_YELLOW}Mid-freq pool size:  {len(mid_pool)}{Style.RESET}")
    print(f"  {Style.FG_YELLOW}Low-freq pool size:  {len(low_pool)}{Style.RESET}")

    return {"high": high_pool, "mid": mid_pool, "low": low_pool}


def generate_query(pools: Dict[str, List[str]], templates: List[str]) -> str:
    """
    About:
    ------
        Generates a single boolean query string by filling in a random template with words from the provided pools.

    Args:
    -----
        pools (Dict[str, List[str]]): A dictionary with keys 'high', 'mid', 'low' mapping to lists of words in each frequency pool.
        templates (List[str]): A list of query templates containing placeholders.

    Returns:
    --------
        A string representing the generated boolean query.
    """
    
    def get_term(pool_name: str) -> str:
        """Helper to get a random, quoted term from a pool."""
        word = random.choice(pools[pool_name])
        return f'"{word}"'

    # Pick a random template
    query = random.choice(templates)

    # Iteratively replace placeholders to ensure different words are used
    while "_H_" in query:
        query = query.replace("_H_", get_term("high"), 1)
    while "_M_" in query:
        query = query.replace("_M_", get_term("mid"), 1)
    while "_L_" in query:
        query = query.replace("_L_", get_term("low"), 1)
    
    return query


# ========================= MAIN =========================
def main():
    """
    About:
    -----
        Main function to generate diverse boolean queries and save them to a JSON file.
    
    Args:
    -----
        None
    
    Returns:
    --------
        None
    """
    parser = argparse.ArgumentParser(description="Generate diverse boolean queries for an information retrieval system.")
    parser.add_argument(
        '-n', '--num_queries', 
        type=int, 
        required=True, 
        help="Number of queries to generate."
    )
    parser.add_argument(
        '-d', '--dataset', 
        type=str, 
        required=True, 
        help="Dataset to use for word frequency calculation (e.g., 'News' or 'Wikipedia')."
    )
    parser.add_argument(
        '-i', '--index_id', 
        type=str, 
        required=True, 
        help="The name/ID of the index these queries are for (e.g., 'my-news-index')."
    )
    parser.add_argument(
        '-o', '--output_file', 
        type=str, 
        required=True, 
        help="Path to the output JSON file to save queries (e.g., 'queries.json')."
    )
    args = parser.parse_args()

    # Get Dataset Handler
    handler = get_dataset_handler(args.dataset)

    # Calculate Word Frequencies
    print(f"\n{Style.FG_CYAN}Calculating word frequencies... (This may take a while){Style.RESET}")
    freq_dict = handler.calculate_word_frequency()
    print(f"{Style.FG_GREEN}Frequency calculation complete.{Style.RESET}")

    # Create Word Pools
    print(f"\n{Style.FG_CYAN}Creating word pools...{Style.RESET}")
    pools = get_word_pools(freq_dict)
    
    # Generate Queries
    print(f"\n{Style.FG_CYAN}Generating {args.num_queries} queries...{Style.RESET}")
    generated_queries = []
    for i in range(args.num_queries):
        query_str = generate_query(pools, QUERY_TEMPLATES)
        generated_queries.append(query_str)
        if (i + 1) % 10 == 0:
            print(f"  Generated {i+1}/{args.num_queries}...")
    
    # Format for JSON output
    output_data = {
        "index_id": args.index_id,
        "n_docs_indexed": -1,
        "preprocessing_settings": None,
        "search_fields": None,
        "attributes_indexed": None,
        "max_results": None,
        "queries": []
    }

    for q_str in generated_queries:
        output_data["queries"].append({
            "query": q_str,
            "docs": []  # Add empty 'docs' list as expected by setup_queries.py
        })

    # Save to file
    try:
        # Add encoding='utf-8' and ensure_ascii=False
        with open(PROJECT_ROOT / args.output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        print(f"\n{Style.FG_GREEN}Successfully saved {args.num_queries} queries to '{args.output_file}'.{Style.RESET}")
    except Exception as e:
        print(f"{Style.FG_RED}Error: Failed to write to file '{args.output_file}'.{Style.RESET}")
        print(f"Details: {e}")


if __name__ == "__main__":
    main()

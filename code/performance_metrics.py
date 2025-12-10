# ======================== IMPORTS ========================
import os
import json
import time
import psutil
import threading
import numpy as np
import seaborn as sns
from tqdm import tqdm
from shared.utils import Style
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from typing import List, Dict, Any, Set
from indexes import ESIndex, CustomIndex, BaseIndex
from dataset_managers import get_news_dataset_handler, get_wikipedia_dataset_handler
from shared.constants import ES_HOST, ES_PORT, ES_SCHEME, OUTPUT_DIR, TEMP_DIR, MAX_NUM_DOCUMENTS, PROJECT_ROOT


# ======================= GLOBALS =========================
memory_usage = []
monitoring = True

print(f"{Style.FG_YELLOW}Using max_num_documents = {MAX_NUM_DOCUMENTS} for performance metrics calculations. To change, modify config.yaml file.{Style.RESET}\n")

INTERVAL = 0.01 # seconds
PLOT_ES = True  # Whether to plot ES index in latency comparison


# ======================= CLASSES =========================
class FunctionalMetrics:
    """
    Class to calculate various functional performance metrics for information retrieval systems.
    """

    def __init__(self):
        pass

    def precision_at_k(self, retrieved: List[str], relevant: Set[str], k: int) -> float:
        """
        About:
        -----
            Calculates Precision@k

        Args:
        -----
            retrieved (List[str]): List of retrieved document IDs.
            relevant (Set[str]): Set of relevant document IDs.
            k (int): The cutoff rank.

        Returns:
        --------
            float: Precision@k value.
        """
        retrieved_at_k = retrieved[:k]
        retrieved_relevant = [doc for doc in retrieved_at_k if doc in relevant]
        return len(retrieved_relevant) / k if k > 0 else 0.0

    def recall_at_k(self, retrieved: List[str], relevant: Set[str], k: int) -> float:
        """
        About:
        -----
            Calculates Recall@k

        Args:
        -----
            retrieved (List[str]): List of retrieved document IDs.
            relevant (Set[str]): Set of relevant document IDs.
            k (int): The cutoff rank.
        
        Returns:
        --------
            float: Recall@k value.
        """

        retrieved_at_k = retrieved[:k]
        retrieved_relevant = [doc for doc in retrieved_at_k if doc in relevant]
        return len(retrieved_relevant) / len(relevant) if len(relevant) > 0 else 0.0

    def f1_score(self, precision: float, recall: float) -> float:
        """
        About:
        -----
            Calculates F1 Score given precision and recall.

        Args:
        -----
            precision (float): Precision value.
            recall (float): Recall value.

        Returns:
        --------
            float: F1 Score value.
        """

        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    def accuracy_at_k(self, retrieved: List[str], relevant: Set[str], k: int, total_docs: int) -> float:
        """
        About:
        -----
            Calculates Accuracy@k

        Args:
        -----
            retrieved (List[str]): List of retrieved document IDs.
            relevant (Set[str]): Set of relevant document IDs.
            k (int): The cutoff rank.
            total_docs (int): Total number of documents in the collection.

        Returns:
        --------
            float: Accuracy@k value.
        """

        if total_docs == 0: return 0.0
        
        retrieved_at_k_set = set(retrieved[:k])
        relevant_set = relevant
        
        tp = len(retrieved_at_k_set.intersection(relevant_set))
        fp = len(retrieved_at_k_set.difference(relevant_set))
        fn = len(relevant_set.difference(retrieved_at_k_set))
        tn = total_docs - (tp + fp + fn)
        
        return (tp + tn) / total_docs if total_docs > 0 else 0.0

    def average_precision(self, retrieved: List[str], relevant: Set[str]) -> float:
        """
        About:
        -----
            Calculates Average Precision (AP)

        Args:
        -----
            retrieved (List[str]): List of retrieved document IDs.
            relevant (Set[str]): Set of relevant document IDs.

        Returns:
        --------
            float: Average Precision value.
        """

        if not relevant:
            return 0.0
        
        ap = 0.0
        num_relevant_hits = 0.0
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                num_relevant_hits += 1
                precision_at_i = num_relevant_hits / (i + 1)
                ap += precision_at_i
                
        return ap / len(relevant) if relevant else 0.0

    def dcg_at_k(self, retrieved: List[str], relevant: Set[str], k: int) -> float:
        """
        About:
        -----
            Calculates Discounted Cumulative Gain (DCG)@k.

        Args:
        -----
            retrieved (List[str]): List of retrieved document IDs.
            relevant (Set[str]): Set of relevant document IDs.
            k (int): The cutoff rank.

        Returns:
        --------
            float: DCG@k value.
        """

        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k]):
            if doc_id in relevant:
                # Using binary relevance (1 if relevant, 0 if not)
                relevance = 1
            else:
                relevance = 0
            dcg += relevance / np.log2(i + 2) # i+2 because log base 2 of 1 (i=0) is 0
        return dcg

    def ndcg_at_k(self, retrieved: List[str], relevant: Set[str], k: int) -> float:
        """
        About:
        -----
            Calculates Normalized Discounted Cumulative Gain (NDCG)@k.

        Args:
        -----
            retrieved (List[str]): List of retrieved document IDs.
            relevant (Set[str]): Set of relevant document IDs.
            k (int): The cutoff rank.

        Returns:
        --------
            float: NDCG@k value.
        """

        # Create the "ideal" ranking: all relevant docs first
        ideal_retrieved = sorted(list(relevant), key=lambda x: x in relevant, reverse=True)
        
        dcg = self.dcg_at_k(retrieved, relevant, k)
        idcg = self.dcg_at_k(ideal_retrieved, relevant, k) # Ideal DCG
        
        return dcg / idcg if idcg > 0 else 0.0


# ======================= THREADS =========================
def monitor_memory(interval: int=1) -> None:
    """
    About:
    ------
        Monitors the system memory usage at regular intervals and appends the usage percentage to the global memory_usage list.

    Args:
    -----
        interval (int): Time interval (in seconds) between memory usage checks.
    
    Returns:
    --------
        None
    """

    while monitoring:
        memory_usage.append(psutil.virtual_memory().percent)
        time.sleep(interval)


# =================== HELPER FUNCTIONS ====================
def clear_folder(folder_path: str) -> None:
    """
    About:
    ------
        Clears all files in the specified folder.

    Args:
    -----
        folder_path (str): Path to the folder to be cleared.
    
    Returns:
    --------
        None
    """

    for file in os.listdir(folder_path):
        os.remove(os.path.join(folder_path, file))

def setup(dataset: str, index_type: str, index_id: str, info: str, dstore: str, qproc: str, compr: str, optim: str, attributes: List[str], query_file_path: str, only_index: bool=False) -> BaseIndex | tuple[BaseIndex, List[dict]]:    
    """
    About:
    -----
        Sets up the dataset handler and index based on the provided parameters. Optionally loads queries from a file.

    Args:
    -----
        dataset (str): The dataset to use ("News" or "Wikipedia").
        index_type (str): The type of index to create ("ESIndex" or "CustomIndex").
        index_id (str): The unique identifier for the index.
        info (str): Index information type.
        dstore (str): Data store type.
        qproc (str): Query processing type.
        compr (str): Compression type.
        optim (str): Optimization type.
        attributes (List[str]): List of attributes to extract from the dataset files.
        query_file_path (str): Path to the query file.
        only_index (bool): If True, only creates and returns the index without loading queries.

    Returns:
    --------
        BaseIndex | tuple[BaseIndex, List[dict]]: The created index instance, and optionally the list of queries if only_index is False.
    """
    
    # Return only the index if specified
    if dataset == "News":
        dataset_handler = get_news_dataset_handler(MAX_NUM_DOCUMENTS, verbose=False)
    elif dataset == "Wikipedia":
        dataset_handler = get_wikipedia_dataset_handler(MAX_NUM_DOCUMENTS, verbose=False)
    else:
        print(f"{Style.FG_RED}Invalid dataset for latency calculation.{Style.RESET}")
        return
    
    # Select Index
    if index_type == "ESIndex":
        index = ESIndex(ES_HOST, ES_PORT, ES_SCHEME, index_type, verbose=False)
    elif index_type == "CustomIndex":
        index = CustomIndex(index_type, info, dstore, qproc, compr, optim)       
    else:
        print(f"{Style.FG_RED}Invalid index type for latency calculation.{Style.RESET}")
        return
    
    # Create Index
    index.create_index(index_id, dataset_handler.get_files(attributes))

    if only_index:
        return index
    
    # Load queries from the query file
    with open(query_file_path, 'r') as f:
        data = json.load(f)

    queries = data.get("queries", [])
    max_results = data.get("max_results", 10)
    
    if not queries:
        print(f"{Style.FG_ORANGE}No queries found in the file.{Style.RESET}")
        return
    
    return index, queries, max_results

def reset() -> None:
    """
    About:
    ------
        Resets the temporary folder by clearing all files in it.
    
    Args:
    -----
        None

    Returns:
    --------
        None
    """

    clear_folder(TEMP_DIR)
    print("\n"+"="*50+"\n")

def get_temp_file_path(metric: str, args: dict) -> str:
    """
    About:
    ------
        Constructs the temporary file path for storing performance metric data based on the provided metric type and arguments.

    Args:
    -----
        metric (str): The type of performance metric (e.g., "memory_usage", "latency").
        args (dict): A dictionary containing the parameters used to create the index and dataset.
    
    Returns:
    --------
        str: The constructed temporary file path.
    """

    return f"{TEMP_DIR}/{metric}_{args.get("index_type")}_{args.get("dataset")}_{args.get("info")}_{args.get("dstore")}_{args.get("qproc")}_{args.get("compr")}_{args.get("optim")}.json"

def get_label(label_fields: List[str]=["index_type"], args: dict={}) -> str:
    """
    About:
    ------
        Constructs a label string based on the specified fields from the provided arguments.

    Args:
    -----
        label_fields (List[str]): List of fields to include in the label.
        args (dict): A dictionary containing the parameters used to create the index and dataset.

    Returns:
        str: The constructed label string.
    """

    label_parts = []
    for label_field in label_fields:
        if label_field not in args:
            raise ValueError(f"Label field '{label_field}' not found in args.")
        if args.get(label_field) and args.get(label_field) != "NONE":
            label_parts.append(args.get(label_field))
    label = "-".join(label_parts) if label_parts else "Custom"
    return label


# ====================== FUNCTIONS ========================
sns.set_theme(style="whitegrid", palette="muted")
sns.set_context("notebook", font_scale=1.1)

def calc_memory_usage_index_creation(index_id: str, index_type: str, dataset: str, info: str, dstore: str, qproc: str, compr: str, optim: str, attributes: List[str]) -> None:
    """
    About:
    ------
        Calculates memory usage during index creation for the specified index and dataset.

    Args:
    -----
        index_id (str): The unique identifier for the index.
        index_type (str): The type of index to create ("ESIndex" or "CustomIndex").
        dataset (str): The dataset to use ("News" or "Wikipedia").
        info (str): Index information type.
        dstore (str): Data store type.
        qproc (str): Query processing type.
        compr (str): Compression type.
        optim (str): Optimization type.
        attributes (List[str]): List of attributes to extract from the dataset files.

    Returns:
    --------
        None
    """
    
    # Reset the variables
    global monitoring, memory_usage
    monitoring = True
    memory_usage.clear()

    # Start monitoring thread
    t = threading.Thread(target=monitor_memory, args=(INTERVAL,))
    t.start()

    index = setup(dataset, index_type, index_id, info, dstore, qproc, compr, optim, attributes, "", only_index=True)

    # Stop monitoring thread
    monitoring = False
    t.join()

    # Delete the created index to free up memory
    index.delete_index(index_id)

def calc_memory_usage_query_execution(query_file_path: str, index_id: str, index_type: str, dataset: str, info: str, dstore: str, qproc: str, compr: str, optim: str, attributes: List[str]) -> None:
    """
    About:
    ------
        Calculates memory usage during query execution for the specified index and dataset.

    Args:
    -----
        query_file_path (str): Path to the query file.
        index_id (str): The unique identifier for the index.
        index_type (str): The type of index to create ("ESIndex" or "CustomIndex").
        dataset (str): The dataset to use ("News" or "Wikipedia").
        info (str): Index information type.
        dstore (str): Data store type.
        qproc (str): Query processing type.
        compr (str): Compression type.
        optim (str): Optimization type.
        attributes (List[str]): List of attributes to extract from the dataset files.

    Returns:
    --------
        None
    """

    # Reset the variables
    global monitoring, memory_usage
    monitoring = True
    memory_usage.clear()

    index, queries, _ = setup(dataset, index_type, index_id, info, dstore, qproc, compr, optim, attributes, PROJECT_ROOT / query_file_path)

    # Start monitoring thread
    t = threading.Thread(target=monitor_memory, args=(INTERVAL,))
    t.start()

    # Execute each query
    index.load_index(index_id)
    for query in queries:
        index.query(query["query"], index_id)
    
    # Stop monitoring thread
    monitoring = False
    t.join()

    # Delete the created index to free up memory
    index.delete_index(index_id)

def plot_memory_usage_comparison(output_file_path: str, args_list: List[str], label_fields: List[str]=["index_type"]) -> None:
    """
    About:
    ------
        Plots memory usage comparison for different index and dataset configurations.

    Args:
    -----
        output_file_path (str): Path to save the output plot image.
        args_list (List[str]): List of argument dictionaries for different index and dataset configurations.
        label_fields (List[str]): List of fields to include in the plot labels.

    Returns:
    --------
        None
    """
    
    datasets = set([args.get("dataset") for args in args_list])
    
    for dataset in datasets:
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Use a color palette
        colors = sns.color_palette("husl", len([args for args in args_list if args.get("dataset") == dataset]))
        color_idx = 0
        
        for args in args_list:
            if args.get("dataset") != dataset:
                continue
            # Load the memory usage data from the corresponding JSON file
            input_file_path: str = get_temp_file_path("memory_usage", args)
            with open(input_file_path, 'r') as f:
                mem_usage_data = json.load(f)

            label = get_label(label_fields, args)
            
            # Plot with seaborn style
            ax.plot(mem_usage_data, label=label, linewidth=2.5, color=colors[color_idx], alpha=0.8)
            color_idx += 1
        
        ax.set_title(f'Memory Usage Comparison on {dataset} Dataset', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Time (seconds)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Memory Usage (%)', fontsize=13, fontweight='bold')
        ax.legend(frameon=True, shadow=True, fancybox=True, fontsize=11, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add subtle background
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('white')
        
        plt.tight_layout()
        output_file_path_db = output_file_path.split(".")[0] + f"_{dataset.lower()}.png"
        plt.savefig(output_file_path_db, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{Style.FG_GREEN}Memory usage comparison plot saved at '{output_file_path_db}'.{Style.RESET}")


def calc_latency(query_file_path: str, index_id: str, index_type: str, dataset: str, info: str, dstore: str, qproc: str, compr: str, optim: str, attributes: List[str]) -> None:
    """
    About:
    ------
        Calculates latency for query execution for the specified index and dataset.

    Args:
    -----
        query_file_path (str): Path to the query file.
        index_id (str): The unique identifier for the index.
        index_type (str): The type of index to create ("ESIndex" or "CustomIndex").
        dataset (str): The dataset to use ("News" or "Wikipedia").
        info (str): Index information type.
        dstore (str): Data store type.
        qproc (str): Query processing type.
        compr (str): Compression type.
        optim (str): Optimization type.
        attributes (List[str]): List of attributes to extract from the dataset files.

    Returns:
    --------
        List[float]: List of latencies for each query execution.
    """
    
    index, queries, _ = setup(dataset, index_type, index_id, info, dstore, qproc, compr, optim, attributes, query_file_path)

    # Execute each query and measure latency
    latency = []
    index.load_index(index_id)
    for query in tqdm(queries, desc="Calculating latency"):
        start_time = time.time()
        index.query(query["query"], index_id)
        end_time = time.time()
        latency.append(end_time - start_time)
    
    # Delete the created index to free up memory
    index.delete_index(index_id)

    return latency

def plot_latency_comparison(output_file_path: str, args_list: List[str], plot_es: bool=True, label_fields: List[str]=["index_type"]) -> None:
    """
    About:
    ------
        Plots latency comparison for different index and dataset configurations.

    Args:
    -----
        output_file_path (str): Path to save the output plot image.
        args_list (List[str]): List of argument dictionaries for different index and dataset configurations.
        plot_es (bool): Whether to include ES index in the plot.
        label_fields (List[str]): List of fields to include in the plot labels.

    Returns:
    --------
        None
    """

    datasets = list(set([args.get("dataset") for args in args_list]))
    
    for dataset in datasets:
        # Collect latency data for this dataset
        latency_data = {}
        for args in args_list:
            # Skip ES plotting if specified
            if not plot_es and args.get("index_type") == "ESIndex":
                continue
            
            if args.get("dataset") != dataset:
                continue
                
            # Load the latency data from the corresponding JSON file
            input_file_path: str = get_temp_file_path("latency", args)
            with open(input_file_path, 'r') as f:
                data = json.load(f)
            latencies = data.get("latencies", [])
            
            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                p95_latency = sorted(latencies)[int(0.95 * len(latencies)) - 1]
                p99_latency = sorted(latencies)[int(0.99 * len(latencies)) - 1]
                
                label = get_label(label_fields, args)
                
                latency_data[label] = (avg_latency, p95_latency, p99_latency)
        
        if not latency_data:
            continue
            
        # Create subplots for the two metrics (removed average)
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Color palette
        colors = sns.color_palette("Set2", len(latency_data))
        x_labels = list(latency_data.keys())
        
        # Plot P95 Latency
        p95_latencies = [latency_data[label][1] for label in x_labels]
        bars1 = axes[0].bar(x_labels, p95_latencies, color=colors, edgecolor='black', linewidth=1.2, alpha=0.8)
        axes[0].set_title('95th Percentile Latency', fontsize=14, fontweight='bold', pad=15)
        axes[0].set_ylabel('Latency (seconds)', fontsize=12, fontweight='bold')
        axes[0].tick_params(axis='x', rotation=45, labelsize=10)
        axes[0].grid(True, alpha=0.3, linestyle='--', axis='y')
        axes[0].set_facecolor('#f8f9fa')
        
        # Add values on top of P95 bars
        for bar, value in zip(bars1, p95_latencies):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(p95_latencies)*0.01, f'{value:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Plot P99 Latency
        p99_latencies = [latency_data[label][2] for label in x_labels]
        bars2 = axes[1].bar(x_labels, p99_latencies, color=colors, edgecolor='black', linewidth=1.2, alpha=0.8)
        axes[1].set_title('99th Percentile Latency', fontsize=14, fontweight='bold', pad=15)
        axes[1].set_ylabel('Latency (seconds)', fontsize=12, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45, labelsize=10)
        axes[1].grid(True, alpha=0.3, linestyle='--', axis='y')
        axes[1].set_facecolor('#f8f9fa')
        
        # Add values on top of P99 bars
        for bar, value in zip(bars2, p99_latencies):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(p99_latencies)*0.01, f'{value:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Set overall title and adjust layout
        fig.suptitle(f'Latency Comparison - {dataset} Dataset', fontsize=16, fontweight='bold', y=1.02)
        fig.patch.set_facecolor('white')
        plt.tight_layout()
        
        # Save plot with dataset-specific filename
        output_file_path_db = output_file_path.replace('.png', f'_{dataset.lower()}.png')
        plt.savefig(output_file_path_db, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{Style.FG_GREEN}Latency comparison plot saved at '{output_file_path_db}'.{Style.RESET}")


def calc_throughput(query_file_path: str, index_id: str, index_type: str, dataset: str, info: str, dstore: str, qproc: str, compr: str, optim: str, attributes: List[str], latencies: List[float]=None) -> None:
    """
    About:
    ------
        Calculates throughput for query execution for the specified index and dataset.

    Args:
    -----
        query_file_path (str): Path to the query file.
        index_id (str): The unique identifier for the index.
        index_type (str): The type of index to create ("ESIndex" or "CustomIndex").
        dataset (str): The dataset to use ("News" or "Wikipedia").
        info (str): Index information type.
        dstore (str): Data store type.
        qproc (str): Query processing type.
        compr (str): Compression type.
        optim (str): Optimization type.
        attributes (List[str]): List of attributes to extract from the dataset files.

    Returns:
    --------
        float: Throughput value (queries per second).
    """
    
    if not latencies:
        index, queries, _ = setup(dataset, index_type, index_id, info, dstore, qproc, compr, optim, attributes, query_file_path)

        # Execute all queries and measure throughput
        start_time = time.time()
        index.load_index(index_id)
        for query in tqdm(queries, desc="Calculating throughput"):
            index.query(query["query"], index_id)
        end_time = time.time()
        total_time = end_time - start_time
        total_queries = len(queries)
    else:
        total_time = sum(latencies)
        total_queries = len(latencies)

    throughput = total_queries / total_time if total_time > 0 else 0

    if not latencies:
        # Delete the created index to free up memory
        index.delete_index(index_id)

    return throughput

def plot_throughput_comparison(output_file_path: str, args_list: List[str], label_fields: List[str]=["index_type"]) -> None:
    """
    About:
    ------
        Plots throughput comparison for different index and dataset configurations.

    Args:
    -----
        output_file_path (str): Path to save the output plot image.
        args_list (List[str]): List of argument dictionaries for different index and dataset configurations.
        label_fields (List[str]): List of fields to include in the plot labels.

    Returns:
    --------
        None
    """
    
    datasets = list(set([args.get("dataset") for args in args_list]))
    for dataset in datasets:
        # Collect throughput data for this dataset
        throughput_data = {}
        for args in args_list:
            if args.get("dataset") != dataset:
                continue
                
            # Load the throughput data from the corresponding JSON file
            input_file_path: str = get_temp_file_path("throughput", args)
            with open(input_file_path, 'r') as f:
                data = json.load(f)
            throughput = data.get("throughput", 0)
            
            if throughput:
                label = get_label(label_fields, args)
                throughput_data[label] = throughput
        
        if not throughput_data:
            continue
            
        # Plot Throughput Comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Color palette
        colors = sns.color_palette("Set3", len(throughput_data))
        x_labels = list(throughput_data.keys())
        throughputs = [throughput_data[label] for label in x_labels]
        
        bars = ax.bar(x_labels, throughputs, color=colors, edgecolor='black', 
                      linewidth=1.2, alpha=0.8)
        ax.set_title(f'Throughput Comparison - {dataset} Dataset', fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('Throughput (queries/second)', fontsize=13, fontweight='bold')
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('white')
        
        # Add values on top of bars
        for bar, value in zip(bars, throughputs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(throughputs)*0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
        plt.tight_layout()
        output_file_path_db = output_file_path.replace('.png', f'_{dataset.lower()}.png')
        plt.savefig(output_file_path_db, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{Style.FG_GREEN}Throughput comparison plot saved at '{output_file_path_db}'.{Style.RESET}")


def calc_functional_metrics(output_dir: str, args_list: List[Dict[str, Any]], label_fields: List[str]=["info", "qproc", "optim"]) -> None:
    """
    About:
    ------
        Calculates functional performance metrics for different index and dataset configurations.

    Args:
    -----
        output_dir (str): Directory to save the output metric files.
        args_list (List[Dict[str, Any]]): List of argument dictionaries for different index and dataset configurations.
        label_fields (List[str]): List of fields to include in the metric labels.

    Returns:
    --------
        None
    """
    
    metrics = FunctionalMetrics()
    all_results_summary = [] # To store dicts for the final table

    for args in args_list:
        print(f"{Style.FG_CYAN}Calculating functional metrics for config: {get_label(label_fields, args)}{Style.RESET}")
        
        # Setup the index and load queries
        # We only run this on CustomIndex, ESIndex is the ground truth
        if args.get("index_type") != "CustomIndex":
            print(f"{Style.FG_ORANGE}Skipping ESIndex, it is the ground truth.{Style.RESET}")
            continue

        try:
            index, queries, max_results = setup(args.get("dataset"),
                                                args.get("index_type"),
                                                args.get("index_id"),
                                                args.get("info"),
                                                args.get("dstore"),
                                                args.get("qproc"),
                                                args.get("compr"),
                                                args.get("optim"),
                                                args.get("attributes"),
                                                PROJECT_ROOT / args.get("query_file_path")
                                                )
            if not index:
                raise Exception("Index setup failed")
        except Exception as e:
            print(f"{Style.FG_RED}Error during setup for {args.get('index_id')}: {e}{Style.RESET}")
            continue

        # Lists to store scores for averaging
        precision_scores, recall_scores, f1_scores, acc_scores, ap_scores, ndcg_scores = [], [], [], [], [], []

        index.load_index(args.get("index_id"))

        # Loop through each query
        for query in queries:
            ground_truth_docs = query['docs'] # Ranked list
            relevant_set = set(ground_truth_docs)
            query_string = query['query']

            # Get retrieved docs from CustomIndex
            try:
                result_json_str = index.query(query_string, args.get("index_id"))
                result_data = json.loads(result_json_str)
                hits = result_data.get("hits", {}).get("hits", [])
                # Get the ranked list of doc IDs from our index
                retrieved_docs = [hit["_id"] for hit in hits]
            except Exception as e:
                print(f"{Style.FG_RED}Error querying index {args.get('index_id')} with query '{query_string}': {e}{Style.RESET}")
                retrieved_docs = []
            
            # Calculate metrics for this query
            p_at_k   : float = metrics.precision_at_k(retrieved_docs, relevant_set, max_results)
            r_at_k   : float = metrics.recall_at_k(retrieved_docs, relevant_set, max_results)
            f1_at_k  : float = metrics.f1_score(p_at_k, r_at_k)
            acc_at_k : float = metrics.accuracy_at_k(retrieved_docs, relevant_set, max_results, MAX_NUM_DOCUMENTS)
            ap       : float = metrics.average_precision(retrieved_docs, relevant_set)
            ndcg_at_k: float = metrics.ndcg_at_k(retrieved_docs, relevant_set, max_results)

            # Append scores for averaging
            precision_scores.append(p_at_k)
            recall_scores.append(r_at_k)
            f1_scores.append(f1_at_k)
            acc_scores.append(acc_at_k)
            ap_scores.append(ap)
            ndcg_scores.append(ndcg_at_k)

        # Average scores for this configuration
        config_summary = {
            "Config": get_label(label_fields, args),
            "Dataset": args.get("dataset"),
            "MAP": np.mean(ap_scores),
            f"NDCG@{max_results}": np.mean(ndcg_scores),
            f"Precision@{max_results}": np.mean(precision_scores),
            f"Recall@{max_results}": np.mean(recall_scores),
            f"F1@{max_results}": np.mean(f1_scores),
            f"Accuracy@{max_results}": np.mean(acc_scores)
        }
        
        all_results_summary.append(config_summary)
        
        # Save detailed results to JSON
        output_file_name = f"metrics_{args.get('dataset')}_{args.get('info')}_{args.get('qproc')}_{args.get('optim')}.json"
        output_file_path = os.path.join(output_dir, output_file_name)
        with open(output_file_path, 'w') as f:
            json.dump(config_summary, f, indent=4)
        print(f"{Style.FG_GREEN}Metrics saved to '{output_file_path}'{Style.RESET}")

        # Clean up index
        index.delete_index(args.get("index_id"))
        reset()

    # Print final summary table
    if not all_results_summary:
        print(f"{Style.FG_RED}No functional metrics were calculated.{Style.RESET}")
        return

    print(f"\n{Style.FG_MAGENTA}===== Functional Metrics Summary ====={Style.RESET}\n")
    
    # Separate tables by dataset
    datasets = sorted(list(set(item['Dataset'] for item in all_results_summary)))
    
    for dataset in datasets:
        print(f"{Style.FG_YELLOW}--- Dataset: {dataset} ---{Style.RESET}")
        table = PrettyTable()
        # Get headers from the first item for this dataset
        headers = [key for key in all_results_summary[0].keys() if key != "Dataset"]
        table.field_names = headers
        
        for col in headers:
            if col != "Config":
                table.float_format[col] = ".4" # Format numbers to 4 decimal places
        
        for result in all_results_summary:
            if result['Dataset'] == dataset:
                table.add_row([result[key] for key in headers])
        
        print(table)
        print("\n")


# ========================= MAIN ==========================
if __name__ == "__main__":
    # Create a temp folder to store intermediate outputs
    os.makedirs(TEMP_DIR, exist_ok=True)

    # ========================================================================
    # Memory footprint comparison for different IndexInfo configurations
    # ========================================================================
    diff_info_mem_args = [
        # ES Index - News Dataset
        {"index_id": "es_news",         "index_type": "ESIndex",     "dataset": "News",      "info": "NONE",      "dstore": "NONE",   "qproc": "NONE", "compr": "NONE", "optim": "NONE", "attributes": ["uuid", "text"]},
        # ES Index - Wikipedia Dataset
        {"index_id": "es_wiki",         "index_type": "ESIndex",     "dataset": "Wikipedia", "info": "NONE",      "dstore": "NONE",   "qproc": "NONE", "compr": "NONE", "optim": "NONE", "attributes": ["id", "text"]},
        # Custom Index - News Dataset
        {"index_id": "cust_news_bool",  "index_type": "CustomIndex", "dataset": "News",      "info": "BOOLEAN",   "dstore": "CUSTOM", "qproc": "NONE", "compr": "NONE", "optim": "NONE", "attributes": ["uuid", "text"]},
        {"index_id": "cust_news_wc",    "index_type": "CustomIndex", "dataset": "News",      "info": "WORDCOUNT", "dstore": "CUSTOM", "qproc": "NONE", "compr": "NONE", "optim": "NONE", "attributes": ["uuid", "text"]},
        {"index_id": "cust_news_tfidf", "index_type": "CustomIndex", "dataset": "News",      "info": "TFIDF",     "dstore": "CUSTOM", "qproc": "NONE", "compr": "NONE", "optim": "NONE", "attributes": ["uuid", "text"]},
        # Custom Index - Wikipedia Dataset
        {"index_id": "cust_wiki_bool",  "index_type": "CustomIndex", "dataset": "Wikipedia", "info": "BOOLEAN",   "dstore": "CUSTOM", "qproc": "NONE", "compr": "NONE", "optim": "NONE", "attributes": ["id", "text"]},
        {"index_id": "cust_wiki_wc",    "index_type": "CustomIndex", "dataset": "Wikipedia", "info": "WORDCOUNT", "dstore": "CUSTOM", "qproc": "NONE", "compr": "NONE", "optim": "NONE", "attributes": ["id", "text"]},
        {"index_id": "cust_wiki_tfidf", "index_type": "CustomIndex", "dataset": "Wikipedia", "info": "TFIDF",     "dstore": "CUSTOM", "qproc": "NONE", "compr": "NONE", "optim": "NONE", "attributes": ["id", "text"]}
    ]
    label_fields = ["index_type", "info"]

    print(f"{Style.FG_MAGENTA}Calculating Memory Usage Comparison for Different Index Information Types...{Style.RESET}")
    for args in diff_info_mem_args:
        print(f"{Style.FG_CYAN}Calculating memory usage for IndexType: {args.get("index_type")}, Dataset: {args.get("dataset")}, Info: {args.get("info")}...{Style.RESET}")
        calc_memory_usage_index_creation(args.get("index_id"), args.get("index_type"), args.get("dataset"), args.get("info"), args.get("dstore"), args.get("qproc"), args.get("compr"), args.get("optim"), args.get("attributes"))

        # Store the memory usage data to a JSON file
        temp_output_file_path: str = get_temp_file_path("memory_usage", args)
        with open(temp_output_file_path, 'w') as f:
            json.dump(memory_usage, f)

    print(f"{Style.FG_MAGENTA}Plotting Memory Usage Comparison for Index: {args.get("index_type")}, Info: {args.get("info")}{Style.RESET}")

    output_file_path: str = os.path.join(OUTPUT_DIR, "diff_info_mem.png")
    plot_memory_usage_comparison(output_file_path, diff_info_mem_args, label_fields)
    reset()


    # ========================================================================
    # Latency comparison for different Data stores
    # ========================================================================
    diff_dstore_latency_args = [
        # ES Index - News Dataset
        {"query_file_path": "query_sets/news_queries.json",      "index_id": "es_news",         "index_type": "ESIndex",     "dataset": "News",      "info": "NONE",    "dstore": "NONE",    "qproc": "NONE", "compr": "NONE", "optim": "NONE", "attributes": ["uuid", "text"]},
        # ES Index - Wikipedia Dataset
        {"query_file_path": "query_sets/wikipedia_queries.json", "index_id": "es_wiki",         "index_type": "ESIndex",     "dataset": "Wikipedia", "info": "NONE",    "dstore": "NONE",    "qproc": "NONE", "compr": "NONE", "optim": "NONE", "attributes": ["id", "text"]},
        # Custom Index - News Dataset
        {"query_file_path": "query_sets/news_queries.json",      "index_id": "cust_news_cust",  "index_type": "CustomIndex", "dataset": "News",      "info": "BOOLEAN", "dstore": "CUSTOM",  "qproc": "NONE", "compr": "NONE", "optim": "NONE", "attributes": ["uuid", "text"]},
        {"query_file_path": "query_sets/news_queries.json",      "index_id": "cust_news_rocks", "index_type": "CustomIndex", "dataset": "News",      "info": "BOOLEAN", "dstore": "ROCKSDB", "qproc": "NONE", "compr": "NONE", "optim": "NONE", "attributes": ["uuid", "text"]},
        {"query_file_path": "query_sets/news_queries.json",      "index_id": "cust_news_redis", "index_type": "CustomIndex", "dataset": "News",      "info": "BOOLEAN", "dstore": "REDIS",   "qproc": "NONE", "compr": "NONE", "optim": "NONE", "attributes": ["uuid", "text"]},
        # Custom Index - Wikipedia Dataset
        {"query_file_path": "query_sets/wikipedia_queries.json", "index_id": "cust_wiki_cust",  "index_type": "CustomIndex", "dataset": "Wikipedia", "info": "BOOLEAN", "dstore": "CUSTOM",  "qproc": "NONE", "compr": "NONE", "optim": "NONE", "attributes": ["id", "text"]},
        {"query_file_path": "query_sets/wikipedia_queries.json", "index_id": "cust_wiki_rocks", "index_type": "CustomIndex", "dataset": "Wikipedia", "info": "BOOLEAN", "dstore": "ROCKSDB", "qproc": "NONE", "compr": "NONE", "optim": "NONE", "attributes": ["id", "text"]},
        {"query_file_path": "query_sets/wikipedia_queries.json", "index_id": "cust_wiki_redis", "index_type": "CustomIndex", "dataset": "Wikipedia", "info": "BOOLEAN", "dstore": "REDIS",   "qproc": "NONE", "compr": "NONE", "optim": "NONE", "attributes": ["id", "text"]}
    ]
    label_fields = ["index_type", "dstore"]

    print(f"{Style.FG_MAGENTA}Calculating Latency of Queries Execution for Different Data Stores...{Style.RESET}")
    for args in diff_dstore_latency_args:
        print(f"{Style.FG_CYAN}Calculating latency for Index: {args.get("index_type")}, DataStore: {args.get("dstore")}, Dataset: {args.get("dataset")}...{Style.RESET}")
        latencies = calc_latency(PROJECT_ROOT / args.get("query_file_path"), args.get("index_id"), args.get("index_type"), args.get("dataset"), args.get("info"), args.get("dstore"), args.get("qproc"), args.get("compr"), args.get("optim"), args.get("attributes"))
        
        # Store the latency data to a JSON file
        temp_output_file_path: str = get_temp_file_path("latency", args)
        with open(temp_output_file_path, 'w') as f:
            json.dump({"latencies": latencies}, f)

    print(f"{Style.FG_MAGENTA}Plotting Latency Comparison for Different Data Stores...{Style.RESET}")

    output_file_path: str = os.path.join(OUTPUT_DIR, "diff_dstore_latency.png")
    plot_latency_comparison(output_file_path, diff_dstore_latency_args, PLOT_ES, label_fields)
    reset()


    # ========================================================================
    # Latency & Throughput comparison for different compression techniques
    # ========================================================================
    diff_compr_latency_throughput_args = [
        # ES Index - News Dataset
        {"query_file_path": "query_sets/news_queries.json",      "index_id": "es_news",        "index_type": "ESIndex",     "dataset": "News",      "info": "NONE",    "dstore": "NONE",   "qproc": "NONE", "compr": "NONE", "optim": "NONE", "attributes": ["uuid", "text"]},
        # ES Index - Wikipedia Dataset
        {"query_file_path": "query_sets/wikipedia_queries.json", "index_id": "es_wiki",        "index_type": "ESIndex",     "dataset": "Wikipedia", "info": "NONE",    "dstore": "NONE",   "qproc": "NONE", "compr": "NONE", "optim": "NONE", "attributes": ["id", "text"]},
        # Custom Index - News Dataset
        {"query_file_path": "query_sets/news_queries.json",      "index_id": "cust_news_none", "index_type": "CustomIndex", "dataset": "News",      "info": "BOOLEAN", "dstore": "CUSTOM", "qproc": "NONE", "compr": "NONE", "optim": "NONE", "attributes": ["uuid", "text"]},
        {"query_file_path": "query_sets/news_queries.json",      "index_id": "cust_news_code", "index_type": "CustomIndex", "dataset": "News",      "info": "BOOLEAN", "dstore": "CUSTOM", "qproc": "NONE", "compr": "CODE", "optim": "NONE", "attributes": ["uuid", "text"]},
        {"query_file_path": "query_sets/news_queries.json",      "index_id": "cust_news_clib", "index_type": "CustomIndex", "dataset": "News",      "info": "BOOLEAN", "dstore": "CUSTOM", "qproc": "NONE", "compr": "CLIB", "optim": "NONE", "attributes": ["uuid", "text"]},
        # Custom Index - Wikipedia Dataset
        {"query_file_path": "query_sets/wikipedia_queries.json", "index_id": "cust_wiki_none", "index_type": "CustomIndex", "dataset": "Wikipedia", "info": "BOOLEAN", "dstore": "CUSTOM", "qproc": "NONE", "compr": "NONE", "optim": "NONE", "attributes": ["id", "text"]},
        {"query_file_path": "query_sets/wikipedia_queries.json", "index_id": "cust_wiki_code", "index_type": "CustomIndex", "dataset": "Wikipedia", "info": "BOOLEAN", "dstore": "CUSTOM", "qproc": "NONE", "compr": "CODE", "optim": "NONE", "attributes": ["id", "text"]},
        {"query_file_path": "query_sets/wikipedia_queries.json", "index_id": "cust_wiki_clib", "index_type": "CustomIndex", "dataset": "Wikipedia", "info": "BOOLEAN", "dstore": "CUSTOM", "qproc": "NONE", "compr": "CLIB", "optim": "NONE", "attributes": ["id", "text"]}
    ]
    label_fields = ["index_type", "compr"]

    print(f"{Style.FG_MAGENTA}Calculating Latency and Throughput of Queries Execution for Different Compression Techniques...{Style.RESET}")
    for args in diff_compr_latency_throughput_args:
        print(f"{Style.FG_CYAN}Calculating latency and throughput for Index: {args.get("index_type")}, Compression: {args.get("compr")}, Dataset: {args.get("dataset")}...{Style.RESET}")
        latencies = calc_latency(PROJECT_ROOT / args.get("query_file_path"),
                                 args.get("index_id"),
                                 args.get("index_type"),
                                 args.get("dataset"),
                                 args.get("info"),
                                 args.get("dstore"),
                                 args.get("qproc"),
                                 args.get("compr"),
                                 args.get("optim"),
                                 args.get("attributes")
                                 )
        throughput = calc_throughput(PROJECT_ROOT / args.get("query_file_path"),
                                     args.get("index_id"),
                                     args.get("index_type"),
                                     args.get("dataset"),
                                     args.get("info"),
                                     args.get("dstore"),
                                     args.get("qproc"),
                                     args.get("compr"),
                                     args.get("optim"),
                                     args.get("attributes"),
                                     latencies
                                     )

        # Store the latency data to a JSON file
        temp_latency_output_file_path: str = get_temp_file_path("latency", args)
        with open(temp_latency_output_file_path, 'w') as f:
            json.dump({"latencies": latencies}, f)

        # Store the throughput data to a JSON file
        temp_throughput_output_file_path: str = get_temp_file_path("throughput", args)
        with open(temp_throughput_output_file_path, 'w') as f:
            json.dump({"throughput": throughput}, f)

    print(f"{Style.FG_MAGENTA}Plotting Latency and Throughput Comparison for Different Compression Techniques...{Style.RESET}")

    output_file_path: str = os.path.join(OUTPUT_DIR, "diff_compr_latency.png")
    plot_latency_comparison(output_file_path, diff_compr_latency_throughput_args, PLOT_ES, label_fields)

    output_file_path: str = os.path.join(OUTPUT_DIR, "diff_compr_throughput.png")
    plot_throughput_comparison(output_file_path, diff_compr_latency_throughput_args, label_fields)
    reset()


    # ========================================================================
    # Memory footprint & latency comparison for Taat/Daat query processing
    # ========================================================================
    diff_qproc_mem_latency_args = [
        # ES Index - News Dataset
        {"query_file_path": "query_sets/news_queries.json",      "index_id": "es_news",        "index_type": "ESIndex",     "dataset": "News",      "info": "NONE",  "dstore": "NONE",   "qproc": "NONE", "compr": "NONE", "optim": "NONE", "attributes": ["uuid", "text"]},
        # ES Index - Wikipedia Dataset
        {"query_file_path": "query_sets/wikipedia_queries.json", "index_id": "es_wiki",        "index_type": "ESIndex",     "dataset": "Wikipedia", "info": "NONE",  "dstore": "NONE",   "qproc": "NONE", "compr": "NONE", "optim": "NONE", "attributes": ["id", "text"]},
        # Custom Index - News Dataset
        {"query_file_path": "query_sets/news_queries.json",      "index_id": "cust_news_taat", "index_type": "CustomIndex", "dataset": "News",      "info": "TFIDF", "dstore": "CUSTOM", "qproc": "TERM", "compr": "NONE", "optim": "NONE", "attributes": ["uuid", "text"]},
        {"query_file_path": "query_sets/news_queries.json",      "index_id": "cust_news_daat", "index_type": "CustomIndex", "dataset": "News",      "info": "TFIDF", "dstore": "CUSTOM", "qproc": "DOC",  "compr": "NONE", "optim": "NONE", "attributes": ["uuid", "text"]},
        # Custom Index - Wikipedia Dataset
        {"query_file_path": "query_sets/wikipedia_queries.json", "index_id": "cust_wiki_taat", "index_type": "CustomIndex", "dataset": "Wikipedia", "info": "TFIDF", "dstore": "CUSTOM", "qproc": "TERM", "compr": "NONE", "optim": "NONE", "attributes": ["id",  "text"]},
        {"query_file_path": "query_sets/wikipedia_queries.json", "index_id": "cust_wiki_daat", "index_type": "CustomIndex", "dataset": "Wikipedia", "info": "TFIDF", "dstore": "CUSTOM", "qproc": "DOC",  "compr": "NONE", "optim": "NONE", "attributes": ["id",  "text"]}
    ]
    label_fields = ["index_type", "qproc"]

    print(f"{Style.FG_MAGENTA}Calculating Memory Usage and Latency of Queries Execution for Different Query Processing Techniques...{Style.RESET}")
    for args in diff_qproc_mem_latency_args:
        print(f"{Style.FG_CYAN}Calculating memory usage and latency for Index: {args.get("index_type")}, Query Processing: {args.get("qproc")}, Dataset: {args.get("dataset")}...{Style.RESET}")
        # Calculate Memory Usage
        calc_memory_usage_query_execution(PROJECT_ROOT / args.get("query_file_path"),
                                          args.get("index_id"),
                                          args.get("index_type"),
                                          args.get("dataset"),
                                          args.get("info"),
                                          args.get("dstore"),
                                          args.get("qproc"),
                                          args.get("compr"),
                                          args.get("optim"),
                                          args.get("attributes")
                                          )

        # Store the memory usage data to a JSON file
        temp_memory_output_file_path: str = get_temp_file_path("memory_usage", args)
        with open(temp_memory_output_file_path, 'w') as f:
            json.dump(memory_usage, f)

        # Calculate Latency
        latencies = calc_latency(PROJECT_ROOT / args.get("query_file_path"),
                                 args.get("index_id"),
                                 args.get("index_type"),
                                 args.get("dataset"),
                                 args.get("info"),
                                 args.get("dstore"),
                                 args.get("qproc"),
                                 args.get("compr"),
                                 args.get("optim"),
                                 args.get("attributes")
                                 )

        # Store the latency data to a JSON file
        temp_latency_output_file_path: str = get_temp_file_path("latency", args)
        with open(temp_latency_output_file_path, 'w') as f:
            json.dump({"latencies": latencies}, f)
    print(f"{Style.FG_MAGENTA}Plotting Memory Usage and Latency Comparison for Different Query Processing Techniques...{Style.RESET}")
    
    output_file_path: str = os.path.join(OUTPUT_DIR, "diff_qproc_mem.png")
    plot_memory_usage_comparison(output_file_path, diff_qproc_mem_latency_args, label_fields)
    
    output_file_path: str = os.path.join(OUTPUT_DIR, "diff_qproc_latency.png")
    plot_latency_comparison(output_file_path, diff_qproc_mem_latency_args, PLOT_ES, label_fields)
    reset()


    # ========================================================================
    # Latency comparison for different optimizations
    # ========================================================================
    diff_optim_latency_args = [
        # ES Index - News Dataset
        {"query_file_path": "query_sets/news_queries.json",      "index_id": "es_news",         "index_type": "ESIndex",     "dataset": "News",      "info": "NONE",  "dstore": "NONE",   "qproc": "NONE", "compr": "NONE", "optim": "NONE",     "attributes": ["uuid", "text"]},
        # ES Index - Wikipedia Dataset
        {"query_file_path": "query_sets/wikipedia_queries.json", "index_id": "es_wiki",         "index_type": "ESIndex",     "dataset": "Wikipedia", "info": "NONE",  "dstore": "NONE",   "qproc": "NONE", "compr": "NONE", "optim": "NONE",    "attributes": ["id", "text"]},
        # Custom Index - News Dataset
        {"query_file_path": "query_sets/news_queries.json",      "index_id": "cust_news_none",  "index_type": "CustomIndex", "dataset": "News",      "info": "TFIDF", "dstore": "CUSTOM", "qproc": "DOC", "compr": "NONE", "optim": "NONE",      "attributes": ["uuid", "text"]},
        {"query_file_path": "query_sets/news_queries.json",      "index_id": "cust_news_optim", "index_type": "CustomIndex", "dataset": "News",      "info": "TFIDF", "dstore": "CUSTOM", "qproc": "DOC", "compr": "NONE", "optim": "OPTIMISED", "attributes": ["uuid", "text"]},
        # Custom Index - Wikipedia Dataset
        {"query_file_path": "query_sets/wikipedia_queries.json", "index_id": "cust_wiki_none",  "index_type": "CustomIndex", "dataset": "Wikipedia", "info": "TFIDF", "dstore": "CUSTOM", "qproc": "DOC", "compr": "NONE", "optim": "NONE",      "attributes": ["id",  "text"]},
        {"query_file_path": "query_sets/wikipedia_queries.json", "index_id": "cust_wiki_optim", "index_type": "CustomIndex", "dataset": "Wikipedia", "info": "TFIDF", "dstore": "CUSTOM", "qproc": "DOC",  "compr": "NONE", "optim": "OPTIMISED", "attributes": ["id",  "text"]}
    ]
    label_fields = ["index_type", "optim"]

    print(f"{Style.FG_MAGENTA}Calculating Latency of Queries Execution for Different Optimization Techniques...{Style.RESET}")
    for args in diff_optim_latency_args:
        print(f"{Style.FG_CYAN}Calculating latency for Index: {args.get("index_type")}, Optimization: {args.get("optim")}, Dataset: {args.get("dataset")}...{Style.RESET}")
        latencies = calc_latency(PROJECT_ROOT / args.get("query_file_path"),
                                 args.get("index_id"),
                                 args.get("index_type"),
                                 args.get("dataset"),
                                 args.get("info"),
                                 args.get("dstore"),
                                 args.get("qproc"),
                                 args.get("compr"),
                                 args.get("optim"),
                                 args.get("attributes")
                                 )
        
        # Store the latency data to a JSON file
        temp_output_file_path: str = get_temp_file_path("latency", args)
        with open(temp_output_file_path, 'w') as f:
            json.dump({"latencies": latencies}, f)
    print(f"{Style.FG_MAGENTA}Plotting Latency Comparison for Different Optimization Techniques...{Style.RESET}")
    output_file_path: str = os.path.join(OUTPUT_DIR, "diff_optim_latency.png")
    plot_latency_comparison(output_file_path, diff_optim_latency_args, PLOT_ES, label_fields)
    reset()


    # ========================================================================
    # Functional Metrics Comparison for different hyperparameters
    # ========================================================================
    diff_hyperparam_metrics_args = [
        # News Dataset - TFIDF - TERM
        {"query_file_path": "query_sets/news_queries.json", "index_id": "cust_news_tfidf_taat_none", "index_type": "CustomIndex", "dataset": "News", "info": "TFIDF", "dstore": "CUSTOM", "qproc": "TERM", "compr": "NONE", "optim": "NONE", "attributes": ["uuid", "text"]},
        
        # News Dataset - TFIDF - DOC
        {"query_file_path": "query_sets/news_queries.json", "index_id": "cust_news_tfidf_daat_none", "index_type": "CustomIndex", "dataset": "News", "info": "TFIDF", "dstore": "CUSTOM", "qproc": "DOC", "compr": "NONE", "optim": "NONE", "attributes": ["uuid", "text"]},
        {"query_file_path": "query_sets/news_queries.json", "index_id": "cust_news_tfidf_daat_es",   "index_type": "CustomIndex", "dataset": "News", "info": "TFIDF", "dstore": "CUSTOM", "qproc": "DOC", "compr": "NONE", "optim": "OPTIMISED", "attributes": ["uuid", "text"]},

        # News Dataset - WORDCOUNT - TERM
        {"query_file_path": "query_sets/news_queries.json", "index_id": "cust_news_wc_taat_none", "index_type": "CustomIndex", "dataset": "News", "info": "WORDCOUNT", "dstore": "CUSTOM", "qproc": "TERM", "compr": "NONE", "optim": "NONE", "attributes": ["uuid", "text"]},

        # News Dataset - WORDCOUNT - DOC
        {"query_file_path": "query_sets/news_queries.json", "index_id": "cust_news_wc_daat_none", "index_type": "CustomIndex", "dataset": "News", "info": "WORDCOUNT", "dstore": "CUSTOM", "qproc": "DOC", "compr": "NONE", "optim": "NONE", "attributes": ["uuid", "text"]},
        {"query_file_path": "query_sets/news_queries.json", "index_id": "cust_news_wc_daat_es",   "index_type": "CustomIndex", "dataset": "News", "info": "WORDCOUNT", "dstore": "CUSTOM", "qproc": "DOC", "compr": "NONE", "optim": "OPTIMISED", "attributes": ["uuid", "text"]},
    ]
    label_fields = ["info", "qproc", "optim"]

    # Run the functional metrics calculation
    print(f"{Style.FG_MAGENTA}Calculating Functional Metrics for Different Hyperparameters...{Style.RESET}")
    calc_functional_metrics(OUTPUT_DIR, diff_hyperparam_metrics_args, label_fields)
    reset()

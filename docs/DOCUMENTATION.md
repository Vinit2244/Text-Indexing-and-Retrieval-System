# Project Documentation: Indexing and Retrieval System

## Table of Contents

1. [Overview](#1-overview)
2. [Features and Implementation](#2-features-and-implementation)
    * [2.1 System Architecture](#21-system-architecture)
    * [2.2 Configuration Components](#22-configuration-components)
    * [2.3 Query Processing](#23-query-processing)
    * [2.4 Query Set Generation](#24-query-set-generation)
3. [Performance Evaluation](#3-performance-evaluation)
4. [How to Run the Code](#4-how-to-run-the-code)
5. [Analysis](#5-analysis)
6. [Assumptions](#6-assumptions)

---

## 1 Overview

This codebase implements a custom **information retrieval system** from the ground up, designed for modularity and rigorous performance evaluation. The core of the project is the `CustomIndex`, a highly configurable inverted index that allows for systematic benchmarking of different storage backends, compression techniques, and query processing algorithms. Its performance is measured against a standard `ESIndex` implementation, which uses Elasticsearch as a production-grade baseline.

---

## 2 Features and Implementation

### 2.1 System Architecture

The system is built around a common `BaseIndex` class, ensuring a consistent API for creating, managing, and querying indices. This allows for seamless swapping between the custom implementation and the Elasticsearch baseline for comparative analysis.

**A. `index_es.py` (Elasticsearch Baseline)**

* **Purpose:** Provides a standard, production-grade baseline for performance and functional comparisons.
* **Implementation:** The `ESIndex` class serves as a wrapper around the `elasticsearch-py` client, connecting to a running Elasticsearch instance.
* **Indexing:** It utilizes the `helpers.streaming_bulk` API for efficient, chunked document ingestion. It relies on Elasticsearch's powerful built-in features, including its default **BM25 scoring** for ranking and optimized Lucene-based storage.
* **Querying:** The system translates its internal query syntax into an Elasticsearch **`bool` query** structure. `AND` maps to `must`, `OR` to `should`, and `NOT` to `must_not`, allowing it to leverage ES's native query execution engine.

**B. `index_custom.py` (Custom Index)**

* **Purpose:** Implements a custom inverted index from scratch, providing full control over every component of the retrieval pipeline.
* **Storage:** All index data is persisted in the `storage/` directory. Each index is stored in its own folder, containing the inverted index, document store, and metadata.
* **Boolean Query Engine:** It integrates a sophisticated `QueryProcessingEngine` with a recursive descent parser (`QueryParser`). This engine correctly handles complex boolean queries with operator precedence (`PHRASE` \> `NOT` \> `AND` \> `OR`) and parentheses for grouping.
* **Ranked Boolean Queries:** The system uses a unique and robust **two-stage hybrid approach** for ranked retrieval:
    1. **Filtering Stage:** The boolean part of the query is executed first, retrieving the exact set of documents that strictly match the logical constraints.
    2. **Ranking Stage:** This filtered subset of documents is then passed to a ranking function (`TAAT` or `DAAT`), which scores and sorts them based on the positive query terms, combining the precision of boolean retrieval with the relevance of scored ranking.
* **Naming Convention:** The custom index follows the pattern `{name}.{info}{dstore}{compr}{optim}{qproc}`, where each part of the extension encodes a specific configuration choice. This allows for easy identification and management of different experimental setups.

### 2.2 Configuration Components

#### `info`: Information Indexed (`BOOLEAN`, `WORDCOUNT`, `TFIDF`)

The `info` parameter controls the richness of the data stored in each term's posting list, directly impacting ranking capabilities and memory usage.

* **`BOOLEAN`**: Stores only document IDs and term positions. It is the most memory-efficient but supports only boolean matching and phrase queries with no relevance scoring.
* **`WORDCOUNT`**: Adds the term frequency (`count`) to each posting. This enables ranking based on how often a term appears within a document, providing a simple measure of relevance.
* **`TFIDF`**: Stores a pre-calculated **TF-IDF score** in each posting. This score considers both term frequency (TF) and inverse document frequency (IDF), allowing for more sophisticated ranking that promotes terms that are important within a document but rare across the entire collection.

**Analysis**: As expected, there is a direct trade-off between functionality and resource consumption. The memory footprint increases from `BOOLEAN` → `WORDCOUNT` → `TFIDF` as more metadata is stored per posting. However, this increased cost unlocks progressively more powerful ranking capabilities, with `TFIDF` providing the most relevant results.

#### `dstore`: Datastore Choices (`CUSTOM`, `REDIS`, `ROCKSDB`)

The `dstore` parameter allows swapping the storage backend for the inverted index and document store.

* **`CUSTOM`**:
  * **Implementation**: Standard file system I/O. The inverted index is serialized into a single binary file, and documents are stored as individual JSON files in a `documents/` subdirectory.
  * **Pros**: Simple, zero-dependency implementation. Fast for datasets that can be fully loaded into memory.
  * **Cons**: Not optimized for selective reads; the entire index must be loaded at once. Can be slow due to repeated file open/close operations.
* **`REDIS`**:
  * **Implementation**: An in-memory key-value store. It uses Redis Hashes to store documents (`index_id:documents`) and simple keys for the inverted index and metadata.
  * **Pros**: Extremely fast read/write operations due to its in-memory nature, making it ideal for low-latency querying.
  * **Cons**: Requires a running Redis server. Memory consumption is directly proportional to the index size, making it potentially expensive for very large datasets.
* **`ROCKSDB`**:
  * **Implementation**: An embedded, persistent key-value store based on a Log-Structured Merge-Tree (LSM-tree).
  * **Pros**: Excellent write performance and efficient on-disk storage. Does not require loading the entire index into memory, enabling it to handle datasets larger than available RAM.
  * **Cons**: More complex to manage due to file locking mechanisms (a single process can hold a lock at a time). Read performance can be slightly slower than in-memory solutions.

#### `compr`: Compression Methods (`NONE`, `CODE`, `CLIB`)

The `compr` parameter controls the compression algorithm applied to the inverted index to reduce its on-disk size.

* **`NONE`**: No compression. The inverted index is serialized to a JSON string and stored as raw bytes. This serves as a performance and size baseline.
* **`CODE`**: **Position-based Postings Compression**.
  * **Implementation**: This is a custom, two-stage algorithm applied only to the position lists within each posting. First, it applies **Gap Encoding** to convert sorted positions into smaller integers (e.g., `[5, 12, 15]` becomes `[5, 7, 3]`). Then, it applies **Variable-Byte (VByte) Encoding** to these gaps, using fewer bytes for smaller numbers.
  * **Analysis**: This method is highly optimized for compressing sorted integer lists and provides a good balance between compression ratio and decompression speed, as it only processes the largest part of the index data (the position lists).
* **`CLIB`**: **Dictionary-level Compression**.
  * **Implementation**: This method applies standard `zlib` compression (based on the DEFLATE algorithm) to the *entire* inverted index after it has been serialized to a JSON string.
  * **Analysis**: This is a general-purpose approach that typically achieves a high compression ratio. However, it comes with a higher CPU overhead for both compression and decompression, which can impact query latency, as the entire index dictionary must be decompressed before a lookup can be performed.

#### `qproc`: Query Processing (`TAAT`, `DAAT`)

The `qproc` parameter determines the algorithm used for the ranking stage of a query.

* **`TAAT` (Term-at-a-Time)**:
  * **Algorithm**: Iterates through each query term one by one. For each term, it retrieves its posting list and updates the scores for all documents in that list.
  * **Analysis**: This approach is conceptually simple. Its performance can be less efficient if a query contains high-frequency terms, as it requires iterating over very long posting lists.
* **`DAAT` (Document-at-a-Time)**:
  * **Algorithm**: First, it fetches the posting lists for *all* query terms. Then, it iterates through each document that matched the boolean filter. For each document, it calculates its total score by looking up its presence and score in the cached posting lists of all query terms.
  * **Analysis**: This approach can be more efficient, especially when combined with optimizations like Early Stopping, as it completes the scoring for one document at a time. This allows for early termination of documents that are unlikely to make the top-k results.

#### `optim`: Ranking Optimizations (`THRESHOLDING`, `EARLYSTOPPING`)

The `optim` parameter enables performance optimizations during the ranking stage to reduce latency.

* **`THRESHOLDING`**:
  * **Algorithm**: A **score pruning** heuristic. After a document's total score is calculated, it is compared against a fixed minimum (`RANKING_SCORE_THRESHOLD`). If the score is below this value, the document is immediately discarded.
  * **Analysis**: This is a simple and effective way to reduce the number of documents that need to be sorted, especially in queries that match many low-relevance documents. It works with both `TAAT` and `DAAT`.
* **`EARLYSTOPPING`**:
  * **Algorithm**: A **Heap-based Top-k** algorithm, implemented within the `DAAT` process. It maintains a min-heap of size `k` (where `k` is `MAX_RESULTS`). As each document is scored, it is only added to the heap if the heap is not yet full, or if its score is greater than the smallest score currently in the heap.
  * **Analysis**: This is a highly efficient optimization that avoids building and sorting a list of all matched documents. By only keeping track of the best `k` candidates seen so far, it significantly reduces both memory consumption and computational complexity, leading to lower query latencies.

### 2.3 Query Processing

The system features a robust query processing engine with support for various query types.

* **Execution Modes**:
  * **Auto/Config**: To be implemented.
  * **Manual**: A CLI-menu based program for interactive index management.
* **Query Syntax**: The engine supports complex boolean queries with a defined operator precedence (`PHRASE` \> `NOT` \> `AND` \> `OR`) and grouping with parentheses.
  * Example: `("Apple" AND "Banana") OR ("Orange" AND NOT "Grape")`
* **TAAT / DAAT Ranking**:
  * The system implements both Term-At-A-Time (TAAT) and Document-At-A-Time (DAAT) algorithms for ranked retrieval.
  * **Note 1**: Ranking requires the index to be created with `IndexInfo = Wordcount` or `Tfidf`. Using a `Boolean` index for a ranked query will result in an error.
  * **Note 2**: Since TAAT and DAAT algorithms are fundamentally designed for "bag of words" queries (implicit OR operations), they cannot be directly applied to complex boolean queries. To address this limitation, the implementation employs a hybrid approach as described below:
* **Ranked Boolean Queries**: To combine the precision of boolean logic with ranked retrieval, the system uses a two-stage process:
  1. First, the boolean query is processed to retrieve a set of all documents that strictly satisfy the logic.
  2. Then, this subset of documents is passed to the TAAT/DAAT ranking function, which scores and ranks them based on the query terms.

### 2.4 Query Set Generation

To rigorously evaluate the performance and correctness of the information retrieval system, a diverse and representative set of queries is essential. The `generate_queries.py` script is designed to automate the creation of such a query set, ensuring it covers a wide range of complexities, term frequencies, and boolean structures. This approach allows for comprehensive testing of the system's capabilities, from simple term lookups to complex logical reasoning.

#### Query Design Strategy

The query generation process is built on a programmatic, template-based strategy that leverages word frequency analysis to create a balanced and challenging query set. This method is designed to be systematic and reproducible, ensuring that the evaluation is both thorough and fair.

The core of the strategy involves:

1. **Word Frequency Analysis**: The script first calculates the frequency of all words in the target dataset (e.g., News or Wikipedia). This allows for the classification of terms based on their prevalence.
2. **Frequency-Based Word Pools**: Words are categorized into three distinct pools:
      * **High-Frequency (`_H_`)**: Common terms that are not stop words (e.g., ranks 100-500). These terms are likely to appear in many documents, testing the system's ability to handle large postings lists.
      * **Mid-Frequency (`_M_`)**: Moderately common terms (e.g., ranks 501-2000). These represent the bulk of meaningful search terms.
      * **Low-Frequency (`_L_`)**: Rare terms (e.g., ranks 2001-10000). These test the system's ability to find specific and potentially unique documents.
3. **Template-Based Generation**: A predefined list of query templates is used to construct queries. These templates mix and match terms from the different frequency pools and combine them with a variety of boolean operators.

#### Query Categories and System Properties Tested

The generated queries are diverse and designed to test various properties of the retrieval system:

| Query Category | Example Template | System Properties Tested |
| :--- | :--- | :--- |
| **Single Term Queries** | `_M_`, `_L_` | **Basic Retrieval**: Efficiency of term lookup and postings list retrieval. Tests the impact of term frequency on latency. |
| **Simple Conjunctions** | `_M_ AND _M_`, `_L_ AND _L_` | **Intersection Logic**: Performance of the `AND` operator. Tests how the system merges postings lists of varying sizes. |
| **Simple Disjunctions** | `_M_ OR _L_` | **Union Logic**: Performance of the `OR` operator. Tests the system's ability to combine and rank results from different terms. |
| **Negation Queries** | `_M_ AND NOT _H_` | **Exclusion Logic**: Correctness and efficiency of the `NOT` operator. Particularly challenging when the negated term is high-frequency. |
| **Complex Boolean Queries** | `(_M_ AND _M_) OR _L_` | **Operator Precedence & Grouping**: Tests the query parser's ability to correctly handle parentheses and follow `AND`/`OR`/`NOT` precedence. |
| **Mixed Frequency Queries**| `_H_ AND _M_`, `(_H_ OR _M_) AND _L_` | **Real-World Scenarios**: Simulates realistic user queries that combine common and specific terms. Tests the robustness of ranking algorithms. |

#### Justification for Diversity

This programmatic approach ensures a more structured and comprehensive query set than manual creation or simple random word sampling. By combining different term frequencies and boolean structures, the generated set is capable of probing the system's performance and correctness across multiple dimensions:

* **Scalability**: Queries with high-frequency terms test how well the system scales with large postings lists.
* **Efficiency**: The mix of simple and complex queries measures the overhead of the query processing engine.
* **Correctness**: Complex boolean logic with negation and parentheses validates the accuracy of the query parser and execution logic.
* **Ranking Quality**: Queries with multiple terms test the effectiveness of the TAAT/DAAT ranking algorithms in producing relevant results.

Once the queries are generated, the `setup_queries.py` script is used to execute them against a baseline Elasticsearch index. The script saves the top-ranking document IDs returned by Elasticsearch, creating a "ground truth" for evaluating the functional correctness (e.g., MAP, NDCG) of the custom index.

---

## 3 Performance Evaluation

The `performance_metrics.py` script is a comprehensive performance evaluation tool that measures and compares various aspects of the indexing and retrieval system.

### System Metrics (Latency, Throughput, Memory)

* **Methodology**: The script provides functions to automate benchmarking. For system metrics, it runs predefined test suites comparing different index configurations (info types, data stores, compression, etc.).
* **Latency (`calc_latency`)**: Measures response times for individual queries, calculating percentile metrics (P95, P99) for performance analysis.
* **Throughput (`calc_throughput`)**: Calculates queries per second to assess system capacity under load.
* **Memory Usage (`monitor_memory`)**: Uses threading to continuously monitor system resources (`psutil`) during index creation and query execution, capturing memory usage patterns at configurable intervals.

### Functional Metrics (Ranking Quality)

* **Methodology**: The `calc_functional_metrics()` function evaluates the ranking quality of the `CustomIndex` against the Elasticsearch ground truth.
* **Metrics Class**: A dedicated `FunctionalMetrics` class implements standard IR evaluation metrics: **Precision@k**, **Recall@k**, **F1@k**, **Accuracy@k**, **Mean Average Precision (MAP)**, and **Normalized Discounted Cumulative Gain (NDCG@k)**.

### Output and Results

* **Performance Plots**: The script generates separate visualizations for each dataset (News/Wikipedia) using Seaborn and Matplotlib, comparing different configurations side-by-side for latency, throughput, and memory.
* **Functional Metrics Tables**: Produces detailed comparison tables showing MAP, NDCG@k, Precision@k, Recall@k, F1@k, and Accuracy@k scores across different index configurations, formatted using `PrettyTable` for easy readability.
* **JSON Reports**: Saves detailed metric data (both system and functional) to JSON files in the output folder for further analysis and persistence of results.

The script serves as the primary evaluation framework for the project, enabling systematic comparison of the custom implementation against Elasticsearch baselines and providing quantitative evidence for design decisions and optimizations.

---

## 4 How to Run the Code

Please refer to [Readme](../README.md)

---

## 5 Analysis

Please refer to [Report](./REPORT.md)

---

## 6 Assumptions

1. No two index will have the same name (apart from the custom extention generated based on the index settings).
2. The Elasticsearch, Redis, and other required services are running on their default or configured ports.

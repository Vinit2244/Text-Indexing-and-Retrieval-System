# How to setup this repository

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed and running
- [Python 3.8+](https://www.python.org/downloads/) installed

## Directory Structure

```text
.
├── code
│   ├── dataset_managers
│   │   ├── dataset_base.py
│   │   ├── dataset_news.py
│   │   └── dataset_wikipedia.py
│   ├── indexes
│   │   ├── encoder.py
│   │   ├── index_base.py
│   │   ├── index_custom.py
│   │   ├── index_es.py
│   │   └── query_processing
│   │       └── query_processing_engine.py
│   ├── preprocessing
│   │   └── preprocessor.py
│   ├── shared
│   │   ├── constants.py
│   │   └── utils.py
│   ├── download_data.py
│   ├── generate_frequency_plots.py
│   ├── generate_queries.py
│   ├── main.py
│   ├── performance_metrics.py
│   ├── preprocess_data.py
│   └── setup_queries.py
├── data
│   ├── news
│   │   └── ...
│   └── wikipedia
│       └── ...
├── docs
│   ├── DOCUMENTATION.md
│   └── REPORT.md
├── output
│   └── ...
├── query_sets
│   ├── example.json
│   ├── news_queries.json
│   └── wikipedia_queries.json
├── storage
│   └── metadata.yaml
├── .gitignore
├── config.yaml
├── README.md
└── requirements.txt
```

## Basic Setup (For MacOS/Linux)

1. Setup Environment:

  ```shell
  # Create environment
  python3 -m venv env

  # Activate environment
  source ./env/bin/activate

  # Download the requirements
  pip install -r requirements.txt
  ```

2. Start docker deamon:

  ```shell
  sudo dockerd # or open using desktop app
  ```

3. Download a local development installation of elasticsearch using docker:

  ```shell
  curl -fsSL https://elastic.co/start-local | sh

  # Please copy the generated username, password and API key and store them in .env file like given below:
  # USERNAME=...
  # PASSWORD=...
  # API_KEY=...
  ```

  Start the container:

  ```shell
  docker start es-local-dev && docker start kibana-local-settings && docker start kibana-local-dev
  ```

  To stop the container once done:

  ```shell
  docker stop kibana-local-dev && docker stop es-local-dev
  ```

4. Run a local server of Redis:

  ```shell
  docker run -d -p 6379:6379 --name my-redis-server redis:latest
  ```

  Start the container:

  ```shell
  docker start my-redis-server
  ```

  To stop the container once done:

  ```shell
  docker stop my-redis-server
  ```

5. Install the C++ RocksDB Library:

  ```shell
  brew install rocksdb
  ```

---

### Preprocess data & Index News/Wikipedia Dataset in Elasticsearch

```shell
# Move into the `code` folder
cd code

# Download the data
python3 download_data.py

# Generate the word frequency plots (before preprocessing)
python3 generate_frequency_plots.py --data_state raw # raw denotes the data has not yet been preprocessed

# Update the preprocessing settings in the config.yaml file before applying preprocessing

# Preprocess the data
python3 preprocess_data.py

# Generate the word frequency plots again
python3 generate_frequency_plots.py --data_state preprocessed # now the data has been preprocessed

# Index News & Wikipedia Data in elasticsearch
python3 main.py # To run in auto mode add the flag "--mode auto". But manual mode is recommended as it gives you more control over indexing your data
# ex: 1 (ESIndex) -> 1 (Create Index) -> 1 (News) -> news_index (Index name) -> uuid, text (Attributes)
# ex: 1 (ESIndex) -> 1 (Create Index) -> 2 (Wikipedia) -> wikipedia_index (Index name) -> id, text (Attributes)
```

### Custom indexing

```shell
# Manipulate the indices
python3 main.py # To run in auto mode (from config file) add the flag "--mode config". But manual mode is recommended as it gives you more control over indexing your data

# Generate queries (Make sure that you run the main.py file and create the index from the data before proceeding as generating queries requires prebuild index)
python3 generate_queries.py -n <N_QUERIES> -d <DATASET> -i <INDEX_ID> -o <PATH_TO_OUTPUT_JSON_FILE>
# python3 generate_queries.py -n 100 -d News -i news_index -o query_sets/news_queries.json
# python3 generate_queries.py -n 100 -d Wikipedia -i wikipedia_index -o query_sets/wikipedia_queries.json

# Generate the query outputs from elasticsearch to compare against custom index
python3 setup_queries.py --path <PATH_TO_QUERIES_JSON> --attributes <ATTR1> <ATTR2> # Multiple spaced attributes on which the dataset was indexed
# ex: python3 setup_queries.py --path query_sets/news_queries.json --attributes uuid text
# ex: python3 setup_queries.py --path query_sets/wikipedia_queries.json --attributes id text

# Performance Testing
python3 performance_metrics.py
```

## About Datasets

### News Dataset

- Source: [news-dataset](https://github.com/Webhose/free-news-datasets)
- Approx. Size: 1.02 GB
- File Type(s): JSON (compressed .json / .json.gz in repository)

### Wikipedia Dataset (English)

- Source: [wikipedia-dataset](https://huggingface.co/datasets/wikimedia/wikipedia/tree/main/20231101.en)
- Approx. Size: 11.6 GB
- File Type(s): Parquet (.parquet)

## Notes

- Wikipedia dataset:
  - Need to manually downaload the .parquet files as downloading via code was leading to corrupt files
- Frequency Plots:
  - Only considers the `text` section of a document
  - Omits punctuations
  - Case insensitive
- Indexes cannot have same names (even when extension is different)
- Queries run will be preprocessed according to how the data is preprocessed (will implement preprocessing queries internally later)

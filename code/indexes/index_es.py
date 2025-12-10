# ======================== IMPORTS ========================
import json
import inspect
from typing import Iterable
from shared.utils import Style
from .index_base import BaseIndex
from elasticsearch import Elasticsearch, helpers
from .query_processing import QueryProcessingEngine
from shared.constants import StatusCode, SEARCH_FIELDS, MAX_RESULTS, USERNAME, PASSWORD, CHUNK_SIZE

print(f"{Style.FG_YELLOW}Using \n\tChunk size: {CHUNK_SIZE}, \n\tMax results: {MAX_RESULTS}, \n\tSearch field: {SEARCH_FIELDS}{Style.RESET}. \nTo change, modify config.yaml file.\n")


# ======================== CLASSES ========================
class ESIndex(BaseIndex):
    """
    Elasticsearch index implementation of the BaseIndex.
    """

    def __init__(self, host: str, port: int, scheme: str, core: str, info: str="NONE", dstore: str="NONE", qproc: str="NONE", compr: str="NONE", optim: str="NONE", verbose: bool=True):
        super().__init__(core, info, dstore, qproc, compr, optim)
        self.host = host
        self.port = port
        self.scheme = scheme
        self.core = core
        status = self._connect_to_cluster(verbose=verbose)

        if status != StatusCode.SUCCESS:
            raise ConnectionError("Failed to connect to Elasticsearch cluster.")
    
    def _connect_to_cluster(self, verbose: bool) -> StatusCode:
        """
        About:
        ------
            Connects to the Elasticsearch cluster.

        Args:
        -----
            verbose (bool): Whether to print status messages.

        Returns:
        --------
            StatusCode: SUCCESS if connection is successful, otherwise CONNECTION_FAILED.
        """

        self.es_client = Elasticsearch(
            [{'host': self.host, 'port': self.port, 'scheme': self.scheme}],
            basic_auth=(USERNAME, PASSWORD)
        )
        try:
            cluster_info = self.es_client.info()
            if verbose:
                print(f"{Style.FG_CYAN}Connected to cluster: {cluster_info['cluster_name']}{Style.RESET}")
                print(f"{Style.FG_CYAN}Elasticsearch version: {cluster_info['version']['number']}{Style.RESET}")
            return StatusCode.SUCCESS
        except Exception as e:
            print(f"{Style.FG_RED}Connection failed: {e}{Style.RESET}")
            return StatusCode.CONNECTION_FAILED

    def get_index_info(self, index_id: str) -> dict | StatusCode:
        """
        About:
        ------
            Retrieves information about the specified index.

        Args:
        -----
            index_id (str): The unique identifier for the index.

        Returns:
        --------
            dict | StatusCode: A dictionary containing index information if successful, otherwise ERROR_ACCESSING_INDEX
        """

        try:
            index_info: dict = self.es_client.indices.get(index=index_id)
            res: dict = self.es_client.count(index=index_id)
            index_info[index_id]["docs_count"] = res["count"]
            return index_info
        
        except Exception as e:
            return StatusCode.ERROR_ACCESSING_INDEX

    def list_indices(self) -> Iterable[str]:
        """
        About:
        ------
            Lists all indices in the Elasticsearch cluster.

        Args:
        -----
            None

        Returns:
        --------
            Iterable[str]: A list of index names.
        """

        return self.es_client.cat.indices(format="json")

    def create_index(self, index_id: str, files: Iterable[tuple[str, dict]]) -> StatusCode:
        """
        About:
        ------
            Creates an index in Elasticsearch and indexes the provided files.

        Args:
        -----
            index_id (str): The unique identifier for the index.
            files (Iterable[tuple[str, dict]]): An iterable of tuples, where each tuple contains the file id and its content.

        Returns:
        --------
            StatusCode: SUCCESS if indexing is successful, otherwise appropriate error code.
        """

        # Create new index if it doesn't exist
        mapping = {
            "mappings": {
                "properties": {
                    MAPPING_FIELD: {  # <-- Use the variable, not "content"
                        "type": "text",
                        "analyzer": "standard",
                        "fields": {
                            "keyword": {"type": "keyword", "ignore_above": 256}
                        }
                    } for MAPPING_FIELD in SEARCH_FIELDS
                    # Add other fields like "title" here if you index them
                }
            }
        }

        if self.es_client.indices.exists(index=index_id):
            return StatusCode.INDEX_ALREADY_EXISTS
        else:
            self.es_client.indices.create(index=index_id, body=mapping)

        # Helper function to resolve if the source is a generator function or an iterable
        def resolve_iterable(source):
            if inspect.isgeneratorfunction(source):
                return source()  # call generator function
            elif hasattr(source, '__iter__'):
                return source
            else:
                raise TypeError("Expected a generator function or iterable.")
        
        # Generator to yield documents in the required format
        def bulk_doc_generator(index_id, files):
            for file in files:
                uuid, doc = file

                yield {
                    "_index": index_id,
                    "_id": uuid,
                    "_source": doc
                }

        failed_flag = False

        # Bulk index documents
        data_iter = resolve_iterable(files)
        for ok, result in helpers.streaming_bulk(client=self.es_client, actions=bulk_doc_generator(index_id, data_iter), chunk_size=CHUNK_SIZE):
            if not ok:
                failed_flag = True
                print("Failed:", result)
        
        if not failed_flag:
            return StatusCode.SUCCESS
        else:
            return StatusCode.INDEXING_FAILED

    def load_index(self, index_id: str) -> StatusCode:
        """
        About:
        ------
            Loads an already created index into memory from disk.

        Args:
        -----
            index_id (str): Id of the index to load

        Returns:
        --------
            StatusCode: SUCCESS if index is accessible, otherwise ERROR_ACCESSING_INDEX.
        """

        # Loading index in Elasticsearch is not required as it manages indices internally
        index_exists = self.get_index_info(index_id)
        if isinstance(index_exists, StatusCode):
            return index_exists
        return StatusCode.SUCCESS

    def update_index(self, index_id: str, remove_files: Iterable[str], add_files: Iterable[tuple[str, dict]]) -> StatusCode:
        """
        About:
        ------
            Updates an index by removing and adding files.

        Args:
        -----
            index_id (str): The unique identifier for the index.
            remove_files (Iterable[str]): An iterable of the file ids to be removed.
            add_files (Iterable[tuple[str, dict]]): An iterable of tuples, where each tuple contains the file id and its content to be added.

        Returns:
        --------
            StatusCode: SUCCESS if update is successful, otherwise appropriate error code.
        """

        if remove_files:
            print(f"{Style.FG_YELLOW}Removing {len(remove_files)} files from index '{index_id}'...{Style.RESET}")
            for uuid in remove_files:
                try:
                    self.es_client.delete(index=index_id, id=uuid)
                except Exception as e:
                    return StatusCode.FAILED_TO_REMOVE_FILE
            
        if add_files:
            print(f"{Style.FG_YELLOW}Adding {len(add_files)} files to index '{index_id}'...{Style.RESET}")
            status = self.create_index(index_id, add_files)
            if status != StatusCode.SUCCESS:
                return status
            
        return StatusCode.SUCCESS

    def query(self, query: str, index_id: str=None) -> str | StatusCode:
        """
        About:
        ------
            Queries the already loaded index to generate a results json and return as str.

        Args:
        -----
            query (str): Input query in str format.
            index_id (str): The unique identifier for the index. If None, searches across all indices.

        Returns:
        --------
            str | StatusCode: Output json str with results if successful, otherwise appropriate error code.
        """

        if index_id is None or index_id.strip() == "":
            index_id = "*"  # Search across all indices if no specific index is provided

        engine = QueryProcessingEngine()
        res = engine.process_es_query(self.es_client, index_id, query, SEARCH_FIELDS, True)

        if isinstance(res, StatusCode):
            return res

        return json.dumps(res.body, indent=2)

    def delete_index(self, index_id: str) -> StatusCode:
        """
        About:
        ------
            Deletes the index with the given index_id.

        Args:
        -----
            index_id (str): The unique identifier for the index.

        Returns:
        --------
            StatusCode: SUCCESS if deletion is successful, otherwise ERROR_ACCESSING_INDEX.
        """

        res = self.es_client.indices.delete(index=index_id.lower())
        if res.get('acknowledged', False):
            return StatusCode.SUCCESS
        else:
            return StatusCode.ERROR_ACCESSING_INDEX

    def list_indexed_files(self, index_id: str) -> Iterable[str] | StatusCode:
        """
        About:
        ------
            Lists all files indexed in the given index.

        Args:
        -----
            index_id (str): The unique identifier for the index.

        Returns:
        --------
            Iterable[str] | StatusCode: A list-like object of file ids if successful, otherwise ERROR_ACCESSING_INDEX.
        """

        # Stream all documents
        try:
            results = helpers.scan(
                client=self.es_client,
                index=index_id,
                query={"query": {"match_all": {}}},
                scroll="5m",
                size=1000  # batch size per scroll request
            )

            results = [doc["_id"] for doc in results]

            return results
        except:
            return StatusCode.ERROR_ACCESSING_INDEX

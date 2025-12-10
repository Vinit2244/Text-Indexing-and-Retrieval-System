# ======================== IMPORTS ========================
import os
import yaml
import json
import math
import zlib
import redis
import shutil
from pathlib import Path
from typing import Iterable
from .encoder import Encoder
from rocksdict import Rdict
from shared.utils import Style
from .index_base import BaseIndex
from .query_processing import QueryProcessingEngine
from shared.constants import IndexInfo, DataStore, Compression, QueryProc, Optimizations, StatusCode, SEARCH_FIELDS, STORAGE_DIR, REDIS_HOST, REDIS_PORT, REDIS_DB


# ======================= GLOBALS ========================
def load_metadata() -> dict:
    file_path = Path(os.path.join(STORAGE_DIR, "metadata.yaml"))

    if not file_path.exists():
        # Create the file with initial content if it doesn't exist
        initial_metadata = {
            "indices": []
        }
        with open(file_path, "w") as f:
            yaml.dump(initial_metadata, f, default_flow_style=False)
        return initial_metadata
    else:
        # Open the existing file for reading and potentially modification
        with open(file_path, "r") as f:
            return yaml.safe_load(f)

METADATA: dict = load_metadata()


# ======================== CLASSES ========================
class CustomIndex(BaseIndex):
    """
    Custom Index class supporting various configurations for indexing and data storage.
    """

    def __init__(self, core: str, info: str="BOOLEAN", dstore: str="CUSTOM", qproc: str="NONE", compr: str="NONE", optim: str="NONE"):
        super().__init__(core, info, dstore, qproc, compr, optim)
        self.core = core
        self.info = info
        self.dstore = dstore
        self.qproc = qproc
        self.compr = compr
        self.encoder = Encoder()
        self.optim = optim
        self.loaded_index = None

        self.name_ext = f"{IndexInfo[info].value}{DataStore[dstore].value}{Compression[compr].value}{Optimizations[optim].value}{QueryProc[qproc].value}"

        # Redis
        self.redis_client = None
        if self.dstore == DataStore.REDIS.name:
            try:
                self.redis_client = redis.Redis(
                    host=REDIS_HOST,
                    port=REDIS_PORT,
                    db=REDIS_DB,
                    decode_responses=True
                )
                self.redis_client.ping() # Test connection
            except Exception as e:
                print(f"Failed to connect to Redis: {e}")
                self.redis_client = None

        # RocksDB
        self.index_db_handle: Rdict = None
        self.doc_store_handle: Rdict = None

    # Private Methods
    def _add_ext_to_index_id(self, index_id: str) -> str | StatusCode:
        """
        About:
        ------
            Adds the appropriate extension to the index_id based on existing indices.

        Args:
        -----
            index_id: The base index identifier.

        Returns:
        --------
            The full index identifier with extension, or StatusCode.INDEX_NOT_FOUND if not found.
        """

        if "." in index_id:
            return index_id
        
        all_indices: list = METADATA.get("indices", [])
        for idx in all_indices:
            if idx.startswith(index_id + "."):
                return idx
        return StatusCode.INDEX_NOT_FOUND

    def _check_index_exists(self, index_id: str) -> bool:
        """
        About:
        ------
            Checks if an index with the given index_id exists.

        Args:
        -----
            index_id: The full index identifier.

        Returns:
        --------
            True if the index exists, False otherwise.
        """

        all_indices: list = METADATA.get("indices", [])
        return index_id in all_indices

    def _get_document(self, index_id_full: str, doc_id: str) -> dict | None:
        """
        About:
        ------
            Retrieves a document's content from the data store based on the index and document ID.

        Args:
        -----
            index_id_full: The full index identifier.
            doc_id: The unique identifier of the document.

        Returns:
        --------
            The document content as a dictionary, or None if not found.
        """

        index_data_path: str = os.path.join(STORAGE_DIR, index_id_full)
        doc_source = None
        try:
            if self.dstore == DataStore.CUSTOM.name:
                doc_path = os.path.join(index_data_path, "documents", f"{doc_id}.json")
                with open(doc_path, "r") as f:
                    doc_source = json.load(f)
            
            elif self.dstore == DataStore.ROCKSDB.name:
                doc_store_path = os.path.join(index_data_path, "doc_store")
                # Use open handle if available (from load_index), else open temp one
                if self.doc_store_handle:
                    doc_bytes = self.doc_store_handle.get(doc_id.encode('utf-8'))
                    if doc_bytes:
                        doc_source = json.loads(doc_bytes.decode('utf-8'))
                else:
                    with Rdict(doc_store_path) as db:
                        doc_bytes = db.get(doc_id.encode('utf-8'))
                        if doc_bytes:
                            doc_source = json.loads(doc_bytes.decode('utf-8'))
            
            elif self.dstore == DataStore.REDIS.name:
                # Use open client if available
                client = self.redis_client
                if not client:
                    # Create a temp client
                    client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
                    client.ping()
                
                doc_json = client.hget(f"{index_id_full}:documents", doc_id)
                if doc_json:
                    doc_source = json.loads(doc_json)
            
            return doc_source

        except Exception as e:
            print(f"{Style.FG_ORANGE}Warning: Could not retrieve document {doc_id} for re-index. Error: {e}{Style.RESET}")
            return None

    def _build_inverted_index(self, files: Iterable[tuple[str, dict]], index_data_path: str, index_id: str) -> dict:
        """
        About:
        ------
            Builds the inverted index from the provided files and stores documents in the data store.

        Args:
        -----
            files: An iterable of tuples, where each tuple contains the file id and its content.
            index_data_path: The path where index data is stored.
            index_id: The unique identifier for the index.

        Returns:
        --------
            A dictionary representing the inverted index.
        """

        # Create place for storing documents if using custom data store
        if self.dstore == DataStore.CUSTOM.name:
            os.mkdir(os.path.join(index_data_path, "documents")) # Folder for storing all the documents
        elif self.dstore == DataStore.ROCKSDB.name:
            doc_store_path: str = os.path.join(index_data_path, "doc_store")
            doc_store = Rdict(doc_store_path)

        inverted_index: dict = {}
        file_count: int = 0

        for uuid, payload in files:
            # Store document in the data store
            if self.dstore == DataStore.CUSTOM.name:
                with open(os.path.join(index_data_path, "documents", f"{uuid}.json"), "w") as f:
                    json.dump(payload, f)
            elif self.dstore == DataStore.ROCKSDB.name:
                doc_store[uuid.encode('utf-8')] = json.dumps(payload).encode('utf-8')
            elif self.dstore == DataStore.REDIS.name:
                if self.redis_client:
                    self.redis_client.hset(f"{index_id}:documents", uuid, json.dumps(payload))

            # Update inverted index
            for search_field in SEARCH_FIELDS:
                content: str = payload.get(search_field, "")
                words: list = content.split()
                for position, word in enumerate(words):
                    if word not in inverted_index:
                        inverted_index[word] = {}
                    if uuid not in inverted_index[word]:
                        inverted_index[word][uuid] = {"positions": []}
                    inverted_index[word][uuid]["positions"].append(position)
                
            file_count += 1
        
        if self.dstore == DataStore.ROCKSDB.name and doc_store:
            doc_store.close()

        # Process based on IndexInfo type
        if self.info == IndexInfo.BOOLEAN.name:
            pass
        
        elif self.info == IndexInfo.WORDCOUNT.name:
            # Store position lists and word counts
            for word in inverted_index:
                for doc_id in inverted_index[word]:
                    positions = inverted_index[word][doc_id]["positions"]
                    inverted_index[word][doc_id]["count"] = len(positions)
        
        elif self.info == IndexInfo.TFIDF.name:
            # Calculate TF-IDF scores
            for word in inverted_index:
                doc_freq = len(inverted_index[word])
                idf = math.log(file_count / doc_freq) if doc_freq > 0 else 0
                
                for doc_id in inverted_index[word]:
                    positions = inverted_index[word][doc_id]["positions"]
                    tf = len(positions)
                    tfidf = tf * idf
                    inverted_index[word][doc_id]["tfidf"] = tfidf

        return inverted_index, file_count

    def _update_index_metadata(self, index_id: str, items: dict) -> dict | StatusCode:
        """
        About:
        ------
            Updates the metadata of the specified index with the provided items.
        
        Args:
        -----
            index_id: The unique identifier for the index.
            items: A dictionary of metadata items to update.

        Returns:
        --------
            None
        """

        index_id = self._add_ext_to_index_id(index_id)
        if index_id == StatusCode.INDEX_NOT_FOUND:
            return StatusCode.INDEX_NOT_FOUND
        
        index_data_path: str = os.path.join(STORAGE_DIR, index_id)

        if self.dstore == DataStore.CUSTOM.name:
            index_metadata_path: str = os.path.join(index_data_path, "metadata.yaml")
            current_info = {}
            if os.path.exists(index_metadata_path):
                with open(index_metadata_path, "r") as f:
                    current_info = yaml.safe_load(f)
            
            current_info.update(items)

            with open(index_metadata_path, "w") as f:
                yaml.dump(current_info, f, default_flow_style=False)
    
        elif self.dstore == DataStore.ROCKSDB.name:
            index_db_path = os.path.join(index_data_path, "index_db")
            try:
                with Rdict(index_db_path) as db:
                    meta_bytes = db.get(b'metadata')
                    current_info = json.loads(meta_bytes.decode('utf-8')) if meta_bytes else {}
                    current_info.update(items)
                    db[b'metadata'] = json.dumps(current_info).encode('utf-8')
            except Exception as e:
                return StatusCode.ERROR_ACCESSING_INDEX
        
        elif self.dstore == DataStore.REDIS.name:
            if not self.redis_client:
                return StatusCode.ERROR_ACCESSING_INDEX
            try:
                key = f"{index_id}:metadata"
                meta_json = self.redis_client.get(key)
                current_info = json.loads(meta_json) if meta_json else {}
                current_info.update(items)
                self.redis_client.set(key, json.dumps(current_info))
            except Exception as e:
                print(f"Redis metadata update error: {e}")
                return StatusCode.ERROR_ACCESSING_INDEX
    
    def _update_global_metadata(self, action: str, index_id: str) -> None:
        """
        About:
        ------
            Updates the global metadata file to add or remove an index entry.

        Args:
        -----
            action: "add" to add the index, "remove" to remove the index.
            index_id: The unique identifier for the index.

        Returns:
        --------
            None
        """

        index_id = self._add_ext_to_index_id(index_id)
        all_indices: list = METADATA.get("indices", [])
        
        if action == "add" and index_id not in all_indices:
            all_indices.append(index_id)
        elif action == "remove" and index_id in all_indices:
            all_indices.remove(index_id)
        
        METADATA["indices"] = all_indices
        with open(os.path.join(STORAGE_DIR, "metadata.yaml"), "w") as f:
            yaml.dump(METADATA, f, default_flow_style=False)

    def _compress(self, inverted_index: dict) -> bytes:
        """
        About:
        ------
            Compresses the inverted index based on the selected compression method.
        
        Args:
        -----
            inverted_index: The inverted index dictionary to compress.
        
        Returns:
        --------
            Compressed inverted index as bytes.
        """

        # CODE Compression
        if self.compr == Compression.CODE.name:
            for term in inverted_index:
                for doc_id in inverted_index[term]:
                    positions = inverted_index[term][doc_id]["positions"]
                    if positions:
                        # Gap encode
                        gaps = self.encoder.gap_encode(positions)
                        # VarByte encode
                        vb_encoded = self.encoder.varbyte_encode(gaps)
                        inverted_index[term][doc_id]["positions"] = vb_encoded.hex() # Store as hex string for JSON compatibility
        
        # Serialize the index into bytes
        data_bytes = json.dumps(inverted_index).encode('utf-8')

        # CLIB Compression
        if self.compr == Compression.CLIB.name:
            data_bytes = zlib.compress(data_bytes)

        return data_bytes

    def _decompress(self, compr: str, data_bytes: bytes) -> dict:
        """
        About:
        ------
            Decompresses the inverted index data based on the specified compression method.

        Args:
        -----
            compr: The compression method used ("NONE", "CLIB", "CODE").
            data_bytes: The compressed inverted index data as bytes.

        Returns:
        --------
            The decompressed inverted index as a dictionary.
        """

        if compr == Compression.CLIB.name:
            data_bytes = zlib.decompress(data_bytes)

        # Deserialize from JSON and CODE decompress
        if self.compr == Compression.CODE.name:
            data_dict = json.loads(data_bytes.decode('utf-8'))
            for term in data_dict:
                for doc_id in data_dict[term]:
                    pos_data = data_dict[term][doc_id]["positions"]

                    if not pos_data:
                        data_dict[term][doc_id]["positions"] = []
                        continue
                    varbyte_blob = bytes.fromhex(pos_data) # Convert hex string back to bytes
                    gaps = self.encoder.varbyte_decode(varbyte_blob)
                    positions = self.encoder.gap_decode(gaps)
                    data_dict[term][doc_id]["positions"] = positions
            return data_dict
        
        else:
            return json.loads(data_bytes.decode('utf-8'))
        
    # Public Methods
    def get_index_info(self, index_id: str) -> dict | StatusCode:
        """
        About:
        ------
            Retrieves metadata information about the specified index.

        Args:
        -----
            index_id: The unique identifier for the index.

        Returns:
        --------
            A dictionary containing index metadata, or StatusCode.ERROR_ACCESSING_INDEX on failure.
        """

        index_id_full = self._add_ext_to_index_id(index_id)
        if not self._check_index_exists(index_id_full):
            return StatusCode.INDEX_NOT_FOUND

        index_data_path: str = os.path.join(STORAGE_DIR, index_id_full)
        index_info = {}

        try:
            if self.dstore == DataStore.CUSTOM.name:
                index_metadata_path: str = os.path.join(index_data_path, "metadata.yaml")
                with open(index_metadata_path, "r") as f:
                    index_info = yaml.safe_load(f)
            
            elif self.dstore == DataStore.ROCKSDB.name:
                index_db_path = os.path.join(index_data_path, "index_db")

                # Check if the handle is already open from load_index()
                if self.index_db_handle:
                    meta_bytes = self.index_db_handle[b'metadata']
                    index_info = json.loads(meta_bytes.decode('utf-8'))
                else:
                    # If index is not loaded, open it temporarily
                    with Rdict(index_db_path) as db:
                        meta_bytes = db[b'metadata']
                        index_info = json.loads(meta_bytes.decode('utf-8'))
            
            elif self.dstore == DataStore.REDIS.name:
                if not self.redis_client:
                    return StatusCode.ERROR_ACCESSING_INDEX
                key = f"{index_id_full}:metadata"
                meta_json = self.redis_client.get(key)
                index_info = json.loads(meta_json)
            
            return {index_id: index_info}

        except Exception as e:
            print(f"Error getting index info: {e}")
            return StatusCode.ERROR_ACCESSING_INDEX

    def list_indices(self) -> Iterable[str]:
        """
        About:
        ------
            Lists all existing indices in the system.

        Args:
        -----
            None

        Returns:
        --------
            An iterable of dictionaries, each containing an index identifier.
        """

        all_indices: list = METADATA.get("indices", [])
        all_indices: list = [{"index": index_id} for index_id in all_indices] # Return list of dicts for consistency with ESIndex
        return all_indices

    def create_index(self, index_id: str, files: Iterable[tuple[str, dict]]) -> StatusCode:
        """
        About:
        ------
            Creates an index for the given files.

        Args:
        -----
            index_id: The unique identifier for the index.
            files: An iterable of tuples, where each tuple contains the file id and its content.

        Returns:
        --------
            StatusCode indicating success or failure.
        """

        index_id = f"{index_id}.{self.name_ext}"
        if self._check_index_exists(index_id):
            return StatusCode.INDEX_ALREADY_EXISTS
        
        index_metadata: dict = {
            "info": self.info,
            "data_store": self.dstore,
            "query_processor": self.qproc,
            "compression": self.compr,
            "optimization": self.optim,
            "documents_indexed": 0
        }

        index_data_path: str = os.path.join(STORAGE_DIR, index_id)

        try:
            # Setup storage location
            if self.dstore == DataStore.CUSTOM.name:
                os.makedirs(index_data_path, exist_ok=True) 
            elif self.dstore == DataStore.ROCKSDB.name:
                os.makedirs(index_data_path, exist_ok=True)
            elif self.dstore == DataStore.REDIS.name:
                if not self.redis_client:
                    return StatusCode.ERROR_ACCESSING_INDEX
                # No directory needed, but we create an empty one for consistency
                os.makedirs(index_data_path, exist_ok=True) 
            
            # Create initial metadata
            self._update_index_metadata(index_id, index_metadata)

            # Build index and store documents
            inverted_index, n_docs_indexed = self._build_inverted_index(files, index_data_path, index_id)

            # Compression
            compressed_inv_idx: bytes = self._compress(inverted_index)
            
            # Save inverted index
            if self.dstore == DataStore.CUSTOM.name:
                with open(os.path.join(index_data_path, "inverted_index.bin"), "wb") as f:
                    f.write(compressed_inv_idx)
            elif self.dstore == DataStore.ROCKSDB.name:
                index_db_path = os.path.join(index_data_path, "index_db")
                with Rdict(index_db_path) as db:
                    db[b'inverted_index'] = compressed_inv_idx
            elif self.dstore == DataStore.REDIS.name:
                self.redis_client.set(f"{index_id}:inverted_index", compressed_inv_idx)

            # Update metadata with document count
            self._update_index_metadata(index_id, {"documents_indexed": n_docs_indexed})
            
            # Update global metadata
            self._update_global_metadata("add", index_id)

            return StatusCode.SUCCESS

        except Exception as e:
            print(f"Error creating index: {e}")
            if os.path.exists(index_data_path):
                shutil.rmtree(index_data_path)
            return StatusCode.ERROR_ACCESSING_INDEX

    def load_index(self, index_id: str) -> StatusCode:
        """
        About:
        ------
            Loads the specified index into memory for querying.

        Args:
        -----
            index_id: The unique identifier for the index.

        Returns:
        --------
            StatusCode indicating success or failure.
        """

        index_id_full = self._add_ext_to_index_id(index_id)
        if not self._check_index_exists(index_id_full):
            return StatusCode.INDEX_NOT_FOUND
        
        index_data_path: str = os.path.join(STORAGE_DIR, index_id_full)

        try:
            index_info = {}
            index_bytes: bytes = None

            # If other RocksDB handles are already open, close them first to prevent a lock conflict on loading.
            if self.index_db_handle:
                self.index_db_handle.close()
                self.index_db_handle = None
            if self.doc_store_handle:
                self.doc_store_handle.close()
                self.doc_store_handle = None

            if self.dstore == DataStore.CUSTOM.name:
                inverted_index_path: str = os.path.join(index_data_path, "inverted_index.bin")
                index_metadata_path: str = os.path.join(index_data_path, "metadata.yaml")

                with open(index_metadata_path, "r") as f:
                    index_info: dict = yaml.safe_load(f)
                with open(inverted_index_path, "rb") as f:
                    index_bytes = f.read()

            elif self.dstore == DataStore.ROCKSDB.name:
                index_db_path = os.path.join(index_data_path, "index_db")
                doc_store_path = os.path.join(index_data_path, "doc_store")
                
                self.index_db_handle = Rdict(index_db_path)
                self.doc_store_handle = Rdict(doc_store_path)
                
                meta_bytes = self.index_db_handle[b'metadata']
                index_info = json.loads(meta_bytes.decode('utf-8'))

                index_bytes = self.index_db_handle[b'inverted_index']
                        
            elif self.dstore == DataStore.REDIS.name:
                if not self.redis_client:
                    return StatusCode.ERROR_ACCESSING_INDEX
                
                meta_json = self.redis_client.get(f"{index_id_full}:metadata")
                index_info = json.loads(meta_json)

                byte_redis = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
                index_bytes = byte_redis.get(f"{index_id_full}:inverted_index")

            if not index_bytes:
                return StatusCode.ERROR_ACCESSING_INDEX
            
            self.loaded_index = self._decompress(index_info.get("compression"), index_bytes)
            
            # Populate self.info from the loaded metadata
            self.info = index_info.get("info", "NONE")
            self.dstore = index_info.get("data_store", "NONE")
            self.qproc = index_info.get("query_processor", "NONE")
            self.compr = index_info.get("compression", "NONE")
            self.optim = index_info.get("optimization", "NONE")
            
            return StatusCode.SUCCESS
        
        except Exception as e:
            # Clean up handles if loading failed, otherwise they stay locked
            if self.index_db_handle:
                self.index_db_handle.close()
                self.index_db_handle = None
            if self.doc_store_handle:
                self.doc_store_handle.close()
                self.doc_store_handle = None
                
            return StatusCode.ERROR_ACCESSING_INDEX

    def update_index(self, index_id: str, remove_files: Iterable[str], add_files: Iterable[tuple[str, dict]]) -> StatusCode:
        """
        About:
        ------
            Updates the specified index by removing and adding files.

        Args:
        -----
            index_id: The unique identifier for the index.
            remove_files: An iterable of file IDs to remove from the index.
            add_files: An iterable of tuples, where each tuple contains the file id and its content

        Returns:
        --------
            StatusCode indicating success or failure.
        """

        # Find the full index ID
        index_id_full = self._add_ext_to_index_id(index_id)
        if index_id_full == StatusCode.INDEX_NOT_FOUND:
            print(f"{Style.FG_RED}Error: Cannot update index '{index_id}' as it does not exist.{Style.RESET}")
            return StatusCode.INDEX_NOT_FOUND
        
        # Get the settings of the current index before we delete it
        # We need this to create the new index with the same properties.
        info_result = self.get_index_info(index_id)
        if isinstance(info_result, StatusCode):
            print(f"{Style.FG_RED}Error: Could not get info for index '{index_id}'. Aborting update.{Style.RESET}")
            return info_result
        index_props = info_result[index_id] 
        
        print(f"{Style.FG_YELLOW}Starting re-index for '{index_id_full}'...{Style.RESET}")

        # Get all current document IDs
        current_doc_ids = self.list_indexed_files(index_id)
        if isinstance(current_doc_ids, StatusCode):
            return current_doc_ids
        
        # Create the new collection of files for re-indexing
        new_file_collection = []
        remove_set = set(remove_files)

        add_dict = {uuid: payload for uuid, payload in add_files}

        print(f"{Style.FG_CYAN}Collecting documents for re-index...{Style.RESET}")
        
        # Add existing files (unless they are in remove_set or add_dict)
        for doc_id in current_doc_ids:
            if doc_id in remove_set:
                continue # Skip, it's marked for deletion
            if doc_id in add_dict:
                continue # Skip, it will be replaced by the (new) version in add_files
            
            # This is a document we need to keep. Fetch its payload.
            payload = self._get_document(index_id_full, doc_id)
            if payload:
                new_file_collection.append((doc_id, payload))
            else:
                print(f"{Style.FG_ORANGE}Warning: Could not find payload for doc_id {doc_id}. It will be dropped from the new index.{Style.RESET}")
        
        # Add all new/updated files
        new_file_collection.extend(add_files)
        
        doc_count = len(new_file_collection)
        if doc_count == 0:
            print(f"{Style.FG_ORANGE}Warning: Update results in an empty index. Deleting index '{index_id}'.{Style.RESET}")
            return self.delete_index(index_id)

        print(f"{Style.FG_CYAN}New index will have {doc_count} documents.{Style.RESET}")

        # Delete the old index. This closes all handles and releases all locks.
        print(f"{Style.FG_CYAN}Deleting old index: {index_id_full}...{Style.RESET}")
        delete_status = self.delete_index(index_id)
        if delete_status != StatusCode.SUCCESS:
            print(f"{Style.FG_RED}Failed to delete old index. Aborting update.{Style.RESET}")
            return delete_status
        
        # Create the new index with the same settings
        print(f"{Style.FG_CYAN}Creating new index: {index_id}...{Style.RESET}")
        
        re_indexer = CustomIndex(
            core=self.core,
            info=index_props.get("info", "BOOLEAN"),
            dstore=index_props.get("data_store", "CUSTOM"),
            qproc=index_props.get("query_processor", "NONE"),
            compr=index_props.get("compression", "NONE"),
            optim=index_props.get("optimization", "NONE")
        )
        
        create_status = re_indexer.create_index(index_id, new_file_collection)
        
        if create_status == StatusCode.SUCCESS:
            print(f"{Style.FG_GREEN}Index '{index_id}' updated successfully.{Style.RESET}")
        else:
            print(f"{Style.FG_RED}Failed to create new index during update.{Style.RESET}")

        return create_status

    def query(self, query: str, index_id: str=None) -> str | StatusCode:
        """
        About:
        ------
            Executes a query against the specified index.

        Args:
        -----
            query: The query string to execute.
            index_id: The unique identifier for the index. If None, uses the core index.

        Returns:
        --------
            A JSON string with the query results formatted like Elasticsearch, or StatusCode on failure.
        """

        if self.loaded_index is None:
            return StatusCode.INDEX_NOT_LOADED

        index_id = self._add_ext_to_index_id(index_id)
        if index_id == StatusCode.INDEX_NOT_FOUND:
            return StatusCode.INDEX_NOT_FOUND

        if not self.loaded_index:
            return StatusCode.ERROR_ACCESSING_INDEX

        engine = QueryProcessingEngine(self.qproc)

        # Get all document IDs (for NOT operations)
        all_doc_ids = self.list_indexed_files(index_id)
        if isinstance(all_doc_ids, StatusCode):
            return all_doc_ids

        matching_doc_ids, hits = engine.process_custom_query(query, self.loaded_index, index_id, all_doc_ids, self.dstore, self.doc_store_handle, self.redis_client, self.info, self.optim)
        if isinstance(matching_doc_ids, StatusCode):
            return matching_doc_ids # Query failed

        # Format output to mimic Elasticsearch
        es_like_output = {
            "hits": {
                "total": {"value": len(matching_doc_ids), "relation": "eq"},
                "hits": hits
            }
        }

        return json.dumps(es_like_output, indent=2)

    def delete_index(self, index_id: str) -> StatusCode:
        """
        About:
        ------
            Deletes the specified index and all its associated data.

        Args:
        -----
            index_id: The unique identifier for the index.

        Returns:
        --------
            StatusCode indicating success or failure.
        """

        index_id = self._add_ext_to_index_id(index_id)
        if not self._check_index_exists(index_id):
            return StatusCode.INDEX_NOT_FOUND

        index_data_path: str = os.path.join(STORAGE_DIR, index_id)
        
        try:
            if self.dstore == DataStore.CUSTOM.name:
                shutil.rmtree(index_data_path) 
            
            elif self.dstore == DataStore.ROCKSDB.name:
                if self.index_db_handle:
                    self.index_db_handle.close()
                    self.index_db_handle = None
                if self.doc_store_handle:
                    self.doc_store_handle.close()
                    self.doc_store_handle = None
                
                # Rdict databases are directories, so just remove the parent
                shutil.rmtree(index_data_path)
            
            elif self.dstore == DataStore.REDIS.name:
                if not self.redis_client:
                    return StatusCode.ERROR_ACCESSING_INDEX
                
                # Find all keys related to this index
                keys_to_delete = []
                for key in self.redis_client.scan_iter(f"{index_id}:*"):
                    keys_to_delete.append(key)
                
                if keys_to_delete:
                    self.redis_client.delete(*keys_to_delete)
                
                # Remove the empty directory
                if os.path.exists(index_data_path):
                    shutil.rmtree(index_data_path)
            
            self._update_global_metadata("remove", index_id)
            return StatusCode.SUCCESS
        
        except Exception as e:
            return StatusCode.ERROR_ACCESSING_INDEX

    def list_indexed_files(self, index_id: str) -> Iterable[str]:
        """
        About:
        ------
            Lists all file IDs indexed in the specified index.

        Args:
        -----
            index_id: The unique identifier for the index.

        Returns:
        --------
            An iterable of file IDs, or StatusCode.INDEX_NOT_FOUND if the index does not exist.
        """

        index_id = self._add_ext_to_index_id(index_id)
        if not self._check_index_exists(index_id):
            return StatusCode.INDEX_NOT_FOUND

        index_data_path: str = os.path.join(STORAGE_DIR, index_id)

        try:
            if self.dstore == DataStore.CUSTOM.name:
                documents_path: str = os.path.join(index_data_path, "documents")
                all_files: list = os.listdir(documents_path)
                all_file_ids: list = [os.path.splitext(filename)[0] for filename in all_files] # Remove .json extension
                return all_file_ids
            
            elif self.dstore == DataStore.ROCKSDB.name:
                doc_store_path = os.path.join(index_data_path, "doc_store")
                if not os.path.exists(doc_store_path):
                    return []
                
                # Use a handle if it's open, otherwise open a temp one
                if self.doc_store_handle:
                    return [key.decode('utf-8') for key in self.doc_store_handle.keys()]
                else:
                    with Rdict(doc_store_path) as db:
                        return [key.decode('utf-8') for key in db.keys()]
            
            elif self.dstore == DataStore.REDIS.name:
                if not self.redis_client:
                    return []
                
                # Get all fields (UUIDs) from the 'documents' hash
                doc_ids = self.redis_client.hkeys(f"{index_id}:documents")
                return doc_ids
        
        except Exception as e:
            print(f"Error listing files: {e}")
            return []

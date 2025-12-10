# ================== IMPORTS ==================
import re
import os
import json
import heapq
from indexes import Encoder
from shared.utils import Style
from typing import List, Dict, Any, Set
from shared.constants import (
    STORAGE_DIR, 
    MAX_RESULTS, 
    DataStore, 
    StatusCode, 
    # Compression,
    IndexInfo,
    QueryProc,
    Optimizations
)


# ==================== GLOBALS =====================
ES_RESULTS_SIZE = 10000  # Max results to fetch from ES for processing


# ================== QUERY PARSER ==================
class QueryParser:
    """
    A simple recursive descent parser for boolean queries with support for AND, OR, NOT, PHRASE, and parentheses.
    """

    def __init__(self, query: str):
        self.tokens = self.tokenize(query)
        self.pos = 0

    def tokenize(self, query: str) -> List[str]:
        """
        About:
        ------
            Tokenizes the input query string into a list of tokens.

        Args:
        -----
            query (str): The input query string.

        Returns:
        -------
            List[str]: A list of tokens extracted from the query.
        """

        # Splits by quotes and operators while keeping phrases
        token_pattern = r'"[^"]+"|\(|\)|AND|OR|NOT|PHRASE'
        tokens = re.findall(token_pattern, query)
        return [t.strip() for t in tokens if t.strip()]

    def peek(self) -> str | None:
        """
        About:
        ------
            Returns the current token without consuming it.

        Args:
        -----
            None

        Returns:
        -------
            str | None: The current token or None if at the end of the token list.
        """
        
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self) -> str:
        """
        About:
        ------
            Consumes and returns the current token, advancing the position.

        Args:
        -----
            None

        Returns:
        -------
            str: The consumed token.
        """

        token = self.peek()
        self.pos += 1
        return token

    def parse(self) -> dict:
        return self.parse_or()

    # Lowest precedence
    def parse_or(self):
        left = self.parse_and()
        while self.peek() == "OR":
            self.consume()
            right = self.parse_and()
            left = {"OR": [left, right]}
        return left

    # Medium precedence
    def parse_and(self):
        left = self.parse_not()
        while self.peek() == "AND":
            self.consume()
            right = self.parse_not()
            left = {"AND": [left, right]}
        return left

    # Next precedence
    def parse_not(self):
        if self.peek() == "NOT":
            self.consume()
            operand = self.parse_phrase()
            return {"NOT": operand}
        return self.parse_phrase()

    # Highest precedence (PHRASE, parentheses, terms)
    def parse_phrase(self):
        token = self.peek()
        if token == "(":
            self.consume()
            expr = self.parse_or()
            assert self.consume() == ")", "Missing closing parenthesis"
            return expr

        if token == "PHRASE":
            self.consume()
            right = self.parse_phrase()
            return {"PHRASE": right}

        term = self.consume()
        if term.startswith('"') and term.endswith('"'):
            return {"TERM": term.strip('"')}

        raise ValueError(f"Unexpected token: {token}")

    # Recursively extract scoring terms (not under NOT)
    def get_scoring_terms(self, node: dict = None) -> List[str]:
        if node is None:
            self.pos = 0 
            node = self.parse()

        if "TERM" in node:
            return [node["TERM"]]
        
        if "PHRASE" in node:
            # Recursively get terms from the phrase
            return self.get_scoring_terms(node["PHRASE"])
        
        if "AND" in node:
            left_terms = self.get_scoring_terms(node["AND"][0])
            right_terms = self.get_scoring_terms(node["AND"][1])
            return left_terms + right_terms
        
        if "OR" in node:
            left_terms = self.get_scoring_terms(node["OR"][0])
            right_terms = self.get_scoring_terms(node["OR"][1])
            return left_terms + right_terms

        if "NOT" in node:
            # Do not return terms under a NOT clause
            return []
        
        if isinstance(node, dict) and not node:
             return []

        raise ValueError(f"Unknown node type in get_scoring_terms: {node}")


# ================== QUERY PROCESSING ENGINE ==================
class QueryProcessingEngine:
    def __init__(self, qproc: str="NONE") -> None:
        self.qproc: str = qproc
        self.encoder = Encoder()

    # Private functions
    def _preprocess_query(self, query: str) -> str:
        return query
    
    def _check_phrase_in_doc(self, doc_id: str, words: list, index: dict) -> bool:
        """
        About:
        ------
            Checks if the given phrase (list of words) exists in the specified document.

        Args:
        -----
            doc_id (str): The ID of the document to check.
            words (list): List of words forming the phrase to check.
            index (dict): The in-memory representation of the index.
        
        Returns:
        --------
            bool: True if the phrase exists in the document, False otherwise.
        """

        def get_positions(word: str, doc_id: str) -> list:
            postings = index.get(word, {}).get(doc_id, {})
            if not postings:
                return []
            
            return postings.get("positions", [])
        
        # Get all positions for the first word in this doc
        first_word_positions = get_positions(words[0], doc_id)
        if not first_word_positions:
            return False

        for pos in first_word_positions:
            # Check if "word2" exists at pos + 1, "word3" at pos + 2, etc.
            match_found = True
            for i, next_word in enumerate(words[1:], start=1):
                next_word_positions = index.get(next_word, {}).get(doc_id, {}).get("positions", [])
                if (pos + i) not in next_word_positions:
                    match_found = False # This sequence failed
                    break
            
            if match_found:
                # Found a full phrase match starting at this position
                return True 
        
        return False # No starting position led to a full match

    def _execute_custom_query(self, node: dict, index: dict, all_docs: set) -> set:
        """
        About:
        ------
            Recursively executes the custom boolean query represented by the parse tree node.

        Args:
        -----
            node (dict): The current node in the parsed query tree.
            index (dict): The in-memory representation of the index.
            all_docs (set): Set of all document IDs in the index.

        Returns:
        --------
            set: A set of document IDs matching the query represented by the node.
        """

        if "TERM" in node:
            term = node["TERM"].lower() # Lowercase to match index
            term_data = index.get(term)
            if term_data:
                return set(term_data.keys()) # Returns {uuid1, uuid2, ...}
            else:
                return set() # Empty set

        if "AND" in node:
            left_docs = self._execute_custom_query(node["AND"][0], index, all_docs)
            right_docs = self._execute_custom_query(node["AND"][1], index, all_docs)
            return left_docs.intersection(right_docs) # AND = intersection

        if "OR" in node:
            left_docs = self._execute_custom_query(node["OR"][0], index, all_docs)
            right_docs = self._execute_custom_query(node["OR"][1], index, all_docs)
            return left_docs.union(right_docs) # OR = union

        if "NOT" in node:
            operand_docs = self._execute_custom_query(node["NOT"], index, all_docs)
            return all_docs.difference(operand_docs) # NOT = set difference

        if "PHRASE" in node:
            inner = node["PHRASE"]
            phrase_term = inner.get("TERM")
            
            if not phrase_term:
                raise ValueError("PHRASE operator must be followed by a single term (e.g., PHRASE \"hello world\")")
            
            words = phrase_term.lower().split() # Lowercase to match index
            if not words:
                return set()
            
            # Get postings for the first word
            first_word = words[0]
            postings = index.get(first_word)
            if not postings:
                return set() # First word not in index

            candidate_docs = set(postings.keys())
            final_docs = set()

            # For each doc, check if the other words follow in sequence
            for doc_id in candidate_docs:
                if self._check_phrase_in_doc(doc_id, words, index):
                    final_docs.add(doc_id)
            
            return final_docs

        raise ValueError(f"Unknown query node: {node}")

    def _get_score_key(self, index_info_type: str) -> str | None:
        """
        About:
        ------
            Maps the index information type to the corresponding score key used in the index postings.

        Args:
        -----
            index_info_type (str): The type of index information (e.g., TFIDF, WORDCOUNT).

        Returns:
        --------
            str | None: The score key corresponding to the index information type, or None if not applicable.
        """

        if index_info_type == IndexInfo.TFIDF.name:
            return "tfidf"
        elif index_info_type == IndexInfo.WORDCOUNT.name:
            return "count"
        else:
            print(f"{Style.FG_RED}Error: Ranking requested but index type '{index_info_type}' is not compatible for ranking. Returning unranked results.{Style.RESET}")
            return None

    def _rank_documents_taat(self, doc_ids_to_rank: Set[str], scoring_terms: List[str], index: Dict[str, Any], index_info_type: str) -> List[Dict[str, Any]]:
        """
        About:
        ------
            Ranks documents using Term-at-a-Time (TAAT) approach.

        Args:
        -----
            doc_ids_to_rank (Set[str]): Set of document IDs to rank.
            scoring_terms (List[str]): List of terms to use for scoring.
            index (Dict[str, Any]): The in-memory index structure.
            index_info_type (str): The type of index information (e.g., TFIDF, WORDCOUNT).

        Returns:
        --------
            List[Dict[str, Any]]: A list of ranked documents with their scores.
        """
        
        score_key = self._get_score_key(index_info_type)

        if not score_key:
            return [{"id": doc_id, "score": 0.0} for doc_id in doc_ids_to_rank]
        
        scores = {} # Accumulator for {doc_id: score}

        # TAAT: Iterate through each scoring term
        for term in scoring_terms:
            term = term.lower()
            if term in index:
                postings = index[term]
                
                # Iterate through all documents for this term
                for doc_id, data in postings.items():
                    if doc_id in doc_ids_to_rank:
                        score = data.get(score_key, 0.0)
                        scores[doc_id] = scores.get(doc_id, 0.0) + score
        
        # Add any matched documents that had no scoring terms
        for doc_id in doc_ids_to_rank:
            if doc_id not in scores:
                scores[doc_id] = 0.0
        
        ranked_list = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        
        return [{"id": doc_id, "score": score} for doc_id, score in ranked_list]

    def _rank_documents_daat(self, doc_ids_to_rank: Set[str], scoring_terms: List[str], index: Dict[str, Any], index_info_type: str, optim: str) -> List[Dict[str, Any]]:
        """
        About:
        ------
            Ranks documents using Document-at-a-Time (DAAT) approach with optional optimizations.
            Optimizations include Early Stopping and Thresholding when 'optim' is set to OPTIMISED.

        Args:
        -----
            doc_ids_to_rank (Set[str]): Set of document IDs to rank.
            scoring_terms (List[str]): List of terms to use for scoring.
            index (Dict[str, Any]): The in-memory index structure.
            index_info_type (str): The type of index information (e.g., TFIDF, WORDCOUNT).
            optim (str): The optimization strategy used (e.g., OPTIMISED, NONE).

        Returns:
        --------
            List[Dict[str, Any]]: A list of ranked documents with their scores.
        """

        score_key = self._get_score_key(index_info_type)
        if not score_key:
            return [{"id": doc_id, "score": 0.0} for doc_id in doc_ids_to_rank]

        # Cache valid terms and their postings
        valid_terms = [t.lower() for t in scoring_terms if t.lower() in index]
        if not valid_terms:
            return [{"id": doc_id, "score": 0.0} for doc_id in doc_ids_to_rank]

        inverted_index = {}
        max_term_scores = {}
        for term in valid_terms:
            postings_dict = index[term]
            postings_list = [
                {"doc_id": d_id, "score": d_data.get(score_key, 0.0)}
                for d_id, d_data in postings_dict.items()
            ]
            postings_list.sort(key=lambda x: x["doc_id"])  # ensure sorted order
            inverted_index[term] = postings_list
            max_term_scores[term] = max((p["score"] for p in postings_list), default=0.0)

        # Optimization flags
        use_optimizations = optim == Optimizations.OPTIMISED.name
        top_k = MAX_RESULTS
        top_k_heap = []  # min-heap
        threshold = 0.0

        # Pointers for each posting list
        pointers = {term: 0 for term in valid_terms}

        for doc_id in sorted(doc_ids_to_rank):
            actual_score = 0.0
            terms_found = set()

            # Step 1: Compute partial score for current doc
            for term in valid_terms:
                postings = inverted_index[term]
                ptr = pointers[term]

                # Advance pointer to doc_id or beyond
                while ptr < len(postings) and postings[ptr]["doc_id"] < doc_id:
                    ptr += 1
                pointers[term] = ptr

                if ptr < len(postings) and postings[ptr]["doc_id"] == doc_id:
                    actual_score += postings[ptr]["score"]
                    terms_found.add(term)

            if use_optimizations:
                # Step 2: Estimate the maximum possible score this document could still reach
                remaining_possible = sum(
                    max_term_scores[t] for t in valid_terms if t not in terms_found
                )
                upper_bound = actual_score + remaining_possible

                # --- Thresholding: skip if even max possible < current threshold
                if len(top_k_heap) >= top_k and upper_bound <= threshold:
                    continue

            # Step 3: Add to heap and maintain threshold
            if len(top_k_heap) < top_k:
                heapq.heappush(top_k_heap, (actual_score, doc_id))
                threshold = top_k_heap[0][0]
            else:
                if actual_score > threshold:
                    heapq.heapreplace(top_k_heap, (actual_score, doc_id))
                    threshold = top_k_heap[0][0]

            # --- Early stopping ---
            if use_optimizations:
                # If remaining documents can’t possibly exceed current threshold
                global_upper_bound = sum(max_term_scores.values())
                if global_upper_bound <= threshold:
                    break

        ranked_list = sorted(top_k_heap, key=lambda item: item[0], reverse=True)
        return [{"id": doc_id, "score": score} for score, doc_id in ranked_list]

    def _build_es_query(self, node, search_fields: list) -> dict:
        """
        About:
        ------
            Recursively builds an Elasticsearch query from the parsed query tree.

        Args:
        -----
            node (dict): The current node in the parsed query tree.
            search_fields (list): List of fields to search in Elasticsearch.

        Returns:
        --------
            dict: The Elasticsearch query representation of the node.
        """
        
        if not node:
            raise ValueError("Query parser returned an empty node.")

        if "TERM" in node:
            # Use "match" for a single term (better relevance than match_phrase)
            return {"multi_match": {
                "query": node["TERM"],
                "fields": search_fields
            }}

        if "AND" in node:
            left, right = node["AND"]
            return {
                "bool": {
                    "must": [self._build_es_query(left, search_fields), self._build_es_query(right, search_fields)]
                }
            }

        if "OR" in node:
            left, right = node["OR"]
            return {
                "bool": {
                    "should": [self._build_es_query(left, search_fields), self._build_es_query(right, search_fields)],
                    "minimum_should_match": 1
                }
            }

        if "NOT" in node:
            return {"bool": {"must_not": [self._build_es_query(node["NOT"], search_fields)]}}

        if "PHRASE" in node:
            inner = node["PHRASE"]
            if "TERM" in inner:
                return {
                    "multi_match": {
                        "query": inner["TERM"],
                        "fields": search_fields,
                        "type": "phrase"
                    }
                }
            return self._build_es_query(inner, search_fields)

        raise ValueError(f"Unknown node type: {node}")

    # Public functions
    def parse_query(self, query: str) -> dict | StatusCode:
        """
        About:
        ------
            Parses the input query string into a parse tree.

        Args:
        -----
            query (str): The input query string.

        Returns:
        -------
            dict | StatusCode: The parse tree representation of the query, or a StatusCode indicating failure.
        """

        try:
            parser = QueryParser(query)
            parsed_tree = parser.parse()
            if parsed_tree is None:
                raise ValueError("Query parser returned None")

        except Exception as e:
            return StatusCode.QUERY_FAILED
        
        return parsed_tree
    
    def process_custom_query(self, query: str, loaded_index: dict, index_id: str, all_doc_ids: List[str], dstore: str, doc_store_handle: Any, redis_client: Any, index_info_type: str, optim: str) -> tuple[set, list] | StatusCode:
        """
        About:
        ------
            Processes a custom boolean query against a loaded index and returns matching document IDs and ranked hits.

        Args:
        -----
            query (str): The boolean query string to process.
            loaded_index (dict): The in-memory representation of the index.
            index_id (str): The unique identifier for the index.
            all_doc_ids (List[str]): List of all document IDs in the index.
            dstore (str): The type of document store used (e.g., CUSTOM, ROCKSDB, REDIS).
            doc_store_handle (Any): Handle to the document store (e.g., RocksDB DB instance).
            redis_client (Any): Redis client instance if using Redis as document store.
            index_info_type (str): The type of index information (e.g., TFIDF, WORDCOUNT).
            optim (str): The optimization strategy used (e.g., OPTIMISED, NONE).

        Returns:
        -------
            tuple[set, list] | StatusCode: A tuple containing the set of all matched document IDs and a list of ranked hits,
            or a StatusCode indicating failure.
        """
        
        # Preprocess the query
        preprocessed_query = self._preprocess_query(query)
        
        try:
            # Parse for boolean execution
            parsed_tree = self.parse_query(preprocessed_query)
            if isinstance(parsed_tree, StatusCode):
                return parsed_tree
            term_parser = QueryParser(preprocessed_query)
            scoring_terms = term_parser.get_scoring_terms()
        
        except Exception as e:
            print(f"{Style.FG_RED}Error during query parsing: {e}{Style.RESET}")
            return StatusCode.QUERY_FAILED

        all_doc_ids_set = set(all_doc_ids)
        if not all_doc_ids_set:
            print(f"{Style.FG_ORANGE}Warning: Index contains no documents.{Style.RESET}")
            all_doc_ids_set = set()

        try:
            matching_doc_ids_set = self._execute_custom_query(parsed_tree, loaded_index, all_doc_ids_set)
        except Exception as e:
            print(f"{Style.FG_RED}Error during query execution: {e}{Style.RESET}")
            return StatusCode.QUERY_FAILED
        
        ranked_docs = []
        
        if self.qproc == QueryProc.TERM.name:
            ranked_docs = self._rank_documents_taat(matching_doc_ids_set, scoring_terms, loaded_index, index_info_type)
        elif self.qproc == QueryProc.DOC.name:
            ranked_docs = self._rank_documents_daat(matching_doc_ids_set, scoring_terms, loaded_index, index_info_type, optim)
        else:
            ranked_docs = [{"id": doc_id, "score": 0.0} for doc_id in matching_doc_ids_set]

        hits = []

        # Iterate over the RANKED list and limit to MAX_RESULTS
        for i, doc_info in enumerate(ranked_docs):
            if i >= MAX_RESULTS:
                break
                
            doc_id = doc_info["id"]
            doc_score = doc_info["score"]
            doc_source = None
            
            try:
                if dstore == DataStore.CUSTOM.name:
                    doc_path = os.path.join(STORAGE_DIR, index_id, "documents", f"{doc_id}.json")
                    with open(doc_path, "r") as f:
                        doc_source = json.load(f)

                elif dstore == DataStore.ROCKSDB.name:
                    # Use the passed-in handle from CustomIndex
                    if doc_store_handle:
                        doc_bytes = doc_store_handle.get(doc_id.encode('utf-8'))
                        if doc_bytes:
                            doc_source = json.loads(doc_bytes.decode('utf-8'))
                
                elif dstore == DataStore.REDIS.name:
                    # Use the passed-in client from CustomIndex
                    if redis_client:
                        doc_json = redis_client.hget(f"{index_id}:documents", doc_id)
                        if doc_json:
                            doc_source = json.loads(doc_json)
                
                if doc_source:
                    hits.append({
                        "_index": index_id,
                        "_id": doc_id,
                        "_score": doc_score, 
                        "_source": doc_source
                    })

            except Exception as e:
                print(f"{Style.FG_ORANGE}Warning: Could not retrieve document {doc_id}. Error: {e}{Style.RESET}")
        
        # Return the original set of all matched IDs, and the formatted/ranked/limited hits
        return matching_doc_ids_set, hits
    
    def process_es_query(self, es_client, index_id: str, query: str, search_fields: list, source: bool=True):
        """
        About:
        ------
            Processes a query using Elasticsearch client.

        Args:
        -----
            es_client: The Elasticsearch client instance.
            index_id (str): The unique identifier for the index.
            query (str): The query string to process.
            search_fields (list): List of fields to search in Elasticsearch.
            source (bool): Whether to include the source document in the results.

        Returns:
        -------
            The search results from Elasticsearch or a StatusCode indicating failure.
        """

        if not search_fields:
            print(f"{Style.FG_RED}Error: No search_fields provided.{Style.RESET}")
            return StatusCode.QUERY_FAILED
        
        # Preprocess the query
        preprocessed_query = self._preprocess_query(query)

        try:
            parser = QueryParser(preprocessed_query)
            parsed_tree = parser.parse()
            
            if parsed_tree is None:
                raise ValueError(f"QueryParser failed to parse query: '{query}'")

            es_query = self._build_es_query(parsed_tree, search_fields)

            res = es_client.search(
                index=index_id, 
                query=es_query, 
                _source=source,
                size=ES_RESULTS_SIZE
            )

            return res

        except Exception as e:
            print(f"{Style.FG_RED}Error during query parsing or execution: {e}{Style.RESET}")
            print(f"Failed query: {query}")
            return StatusCode.QUERY_FAILED

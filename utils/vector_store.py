import os
import uuid
import re
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility

class VectorStore:
    def __init__(self):
        self.host = os.getenv('MILVUS_HOST', 'localhost')
        self.port = os.getenv('MILVUS_PORT', '19530')
        self.connect_to_milvus()

    def connect_to_milvus(self):
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )
            print("✅ Connected to Milvus successfully")
        except Exception as e:
            print(f"❌ Error connecting to Milvus: {str(e)}")
            raise

    def _format_collection_name(self, session_id):
        safe_id = re.sub(r'[^a-zA-Z0-9_]', '', session_id)
        return f"sess_{safe_id}"

    def create_collection_schema(self):
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=5000),
            FieldSchema(name="original_text", dtype=DataType.VARCHAR, max_length=10000),
            FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
        ]
        return CollectionSchema(fields=fields, description="Document embeddings collection")

    def create_collection(self, session_id):
        try:
            collection_name = self._format_collection_name(session_id)
            if utility.has_collection(collection_name):
                return Collection(collection_name)

            schema = self.create_collection_schema()
            collection = Collection(collection_name, schema)

            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": "COSINE",
                "params": {"nlist": 1024}
            }
            collection.create_index(field_name="embedding", index_params=index_params)
            return collection
        except Exception as e:
            print(f"❌ Error creating collection: {str(e)}")
            raise

    def collection_exists(self, session_id):
        try:
            collection_name = self._format_collection_name(session_id)
            return utility.has_collection(collection_name)
        except Exception as e:
            print(f"❌ Error checking collection existence: {str(e)}")
            return False

    def add_documents(self, session_id, documents, filename):
        try:
            collection_name = self._format_collection_name(session_id)
            collection = self.create_collection(session_id)

            ids = []
            texts = []
            original_texts = []
            filenames = []
            embeddings = []

            for doc in documents:
                ids.append(str(uuid.uuid4()))
                texts.append(doc['text'])
                original_texts.append(doc['original_text'])
                filenames.append(filename)
                embeddings.append(doc['embedding'])

            data = [ids, texts, original_texts, filenames, embeddings]
            collection.insert(data)
            collection.flush()
            collection.load()

            print(f"✅ Added {len(documents)} documents to collection {collection_name}")
            return True
        except Exception as e:
            print(f"❌ Error adding documents: {str(e)}")
            return False
    
    def add_codebase(self, session_id, code_chunks, source_name):
        """Add codebase chunks to vector store"""
        try:
            collection_name = self._format_collection_name(session_id)
            collection = self.create_codebase_collection(session_id)

            ids = []
            contents = []
            file_paths = []
            chunk_types = []
            languages = []
            embeddings = []
            metadata = []

            for chunk in code_chunks:
                ids.append(str(uuid.uuid4()))
                contents.append(chunk.get('content', ''))
                file_paths.append(chunk.get('file_path', ''))
                chunk_types.append(chunk.get('chunk_type', 'code'))
                languages.append(chunk.get('language', 'text'))
                embeddings.append(chunk['embedding'])
                
                # Store additional metadata as JSON string
                chunk_metadata = {
                    'start_line': chunk.get('start_line', 0),
                    'end_line': chunk.get('end_line', 0),
                    'source': source_name
                }
                metadata.append(str(chunk_metadata))

            data = [ids, contents, file_paths, chunk_types, languages, embeddings, metadata]
            collection.insert(data)
            collection.flush()
            collection.load()

            print(f"✅ Added {len(code_chunks)} code chunks to collection {collection_name}")
            return True
        except Exception as e:
            print(f"❌ Error adding codebase: {str(e)}")
            return False
    
    def create_codebase_collection(self, session_id):
        """Create collection schema for codebase storage"""
        try:
            collection_name = self._format_collection_name(session_id)
            if utility.has_collection(collection_name):
                return Collection(collection_name)

            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=10000),
                FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="language", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
                FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=1000)
            ]
            
            schema = CollectionSchema(fields=fields, description="Codebase embeddings collection")
            collection = Collection(collection_name, schema)

            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": "COSINE",
                "params": {"nlist": 1024}
            }
            collection.create_index(field_name="embedding", index_params=index_params)
            return collection
        except Exception as e:
            print(f"❌ Error creating codebase collection: {str(e)}")
            raise
    
    def search_codebase(self, session_id, query_text, top_k=5):
        """Search codebase with query"""
        try:
            collection_name = self._format_collection_name(session_id)
            if not utility.has_collection(collection_name):
                return []

            collection = Collection(collection_name)
            collection.load()

            from .codebase_processor import CodebaseProcessor
            codebase_processor = CodebaseProcessor()
            query_embedding = codebase_processor.get_embedding(query_text)

            if not query_embedding:
                return []

            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }

            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["content", "file_path", "chunk_type", "language", "metadata"]
            )

            code_chunks = []
            for result in results[0]:
                code_chunks.append({
                    'content': result.entity.get('content'),
                    'file_path': result.entity.get('file_path'),
                    'chunk_type': result.entity.get('chunk_type'),
                    'language': result.entity.get('language'),
                    'metadata': result.entity.get('metadata'),
                    'score': result.score
                })

            return code_chunks
        except Exception as e:
            print(f"❌ Error searching codebase: {str(e)}")
            return []

    def search_documents(self, session_id, query_text, top_k=5):
        """Search documents with query"""
        try:
            collection_name = self._format_collection_name(session_id)
            if not utility.has_collection(collection_name):
                print(f"Collection {collection_name} does not exist")
                return []

            collection = Collection(collection_name)
            collection.load()

            from .document_processor import DocumentProcessor
            doc_processor = DocumentProcessor()
            query_embedding = doc_processor.get_embedding(query_text)

            if not query_embedding:
                print("Failed to generate query embedding for document search")
                return []

            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }

            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["text", "original_text", "filename"]
            )

            documents = []
            for result in results[0]:
                documents.append({
                    'text': result.entity.get('text'),
                    'original_text': result.entity.get('original_text'),
                    'filename': result.entity.get('filename'),
                    'score': result.score
                })

            print(f"Found {len(documents)} relevant document chunks")
            return documents
        except Exception as e:
            print(f"❌ Error searching documents: {str(e)}")
            return []

    def delete_collection(self, session_id):
        try:
            collection_name = self._format_collection_name(session_id)
            if utility.has_collection(collection_name):
                utility.drop_collection(collection_name)
                print(f"🧹 Deleted collection {collection_name}")
            return True
        except Exception as e:
            print(f"❌ Error deleting collection: {str(e)}")
            return False

    def get_collection_stats(self, session_id):
        try:
            collection_name = self._format_collection_name(session_id)
            if not utility.has_collection(collection_name):
                return None

            collection = Collection(collection_name)
            collection.load()

            return {
                'name': collection_name,
                'num_entities': collection.num_entities,
                'description': collection.description
            }
        except Exception as e:
            print(f"❌ Error getting collection stats: {str(e)}")
            return None
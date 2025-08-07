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

    def search_documents(self, session_id, query_text, top_k=5):
        try:
            collection_name = self._format_collection_name(session_id)
            if not utility.has_collection(collection_name):
                return []

            collection = Collection(collection_name)
            collection.load()

            from .document_processor import DocumentProcessor
            doc_processor = DocumentProcessor()
            query_embedding = doc_processor.get_embedding(query_text)

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

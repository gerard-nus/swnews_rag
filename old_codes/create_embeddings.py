import os
from dotenv import load_dotenv
import jsonlines
from llama_index.core import Document, Settings, VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter  # Updated import
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore


def load_environment():
    """Load environment variables"""
    load_dotenv()
    return {
        'PINECONE_API_KEY': os.getenv('PINECONE_API_KEY'),
        'ENVIRONMENT': os.getenv('ENVIRONMENT', 'gcp-starter')
    }


def setup_embedding_model():
    """Configure the embedding model"""
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L12-v2",
        embed_batch_size=100,
        max_length=512
    )
    return Settings.embed_model


def initialize_pinecone(api_key: str, environment: str):
    """Initialize Pinecone client"""
    return Pinecone(
        api_key=api_key,
        environment=environment
    )


def create_or_get_index(pc, index_name: str, dimension: int = 384):
    """Create Pinecone index if it doesn't exist, or get existing index"""
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric='cosine'
        )
    return pc.Index(index_name)


def process_json_documents(json_path: str):
    """Process documents from JSON file"""
    # Load JSON data
    with jsonlines.open(json_path, "r") as documents:
        data_list = list(documents)

    # Convert to documents with metadata
    documents_with_metadata = [
        Document(
            text=item[0]['text'],
            metadata={
                "title": item[0]['title'],
                "pub_date": item[0]['pub_date'],
                "article_source": item[0].get('article_source', '')
            }
        )
        for item in data_list if isinstance(item, list) and len(item) > 0
    ]

    return documents_with_metadata


def main():
    # Load environment variables
    env_vars = load_environment()

    # Setup embedding model
    embed_model = setup_embedding_model()

    # Initialize Pinecone
    pc = initialize_pinecone(
        api_key=env_vars['PINECONE_API_KEY'],
        environment=env_vars['ENVIRONMENT']
    )

    # Create or get index
    index_name = "socialworknews-index"
    pinecone_index = create_or_get_index(pc, index_name)

    # Create vector store
    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index
    )

    # Create storage context
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    # Process JSON documents
    documents_with_metadata = process_json_documents("data/json/data_rag_json.json")

    # Create text splitter with the correct import
    text_splitter = SentenceSplitter(chunk_size=210, chunk_overlap=50)

    # Create index with Pinecone
    index = VectorStoreIndex.from_documents(
        documents_with_metadata,
        storage_context=storage_context,
        transformations=[text_splitter],
        show_progress=True
    )

    print(f"Successfully processed and stored {len(documents_with_metadata)} documents in Pinecone")


if __name__ == "__main__":
    main()
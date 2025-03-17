import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import ChatPromptTemplate
from loguru import logger

Settings.llm = HuggingFaceLLM(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    tokenizer_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    context_window=2048,
    max_new_tokens=156,
    generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95, "do_sample": True}
)
logger.debug(f"Loaded LLM model: {Settings.llm.model_name}")

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)
logger.debug(f"Loaded embedding model: {Settings.embed_model.model_name}")

documents = SimpleDirectoryReader(input_dir="data/", required_exts=['.pdf']).load_data()
logger.debug(f"Read {len(documents)} documents from data/")

index = VectorStoreIndex.from_documents(documents)
logger.debug("Created VectorStoreIndex")

chat_text_qa_msgs = [
    (
        "user",
        """You are a Q&A assistant. Your goal is to answer questions as
accurately as possible based on the instructions and context provided.

Context:

{context_str}

Question:

{query_str}
"""
    )
]

text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)
query_engine = index.as_query_engine(text_qa_template=text_qa_template)
logger.debug("Created QueryEngine")
logger.info("__________________________________________________________")
logger.info("\n")
query = "What is the goal of the Q&A assistant?"
response = query_engine.query(query)
logger.info(f'Query: {query}')
logger.info(f"Response: {response}")
logger.info("\n")
logger.info("__________________________________________________________")
logger.info("\n")
query = "What is the scientific paper about?"
response = query_engine.query(query)
logger.info(f'Query: {query}')
logger.info(f"Response: {response}")
logger.info("\n")
logger.info("__________________________________________________________")
logger.info("\n")

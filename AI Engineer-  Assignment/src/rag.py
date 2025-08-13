from pathlib import Path
from typing import Iterable, List, Dict, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

try:
    from src.embeddings import build_embeddings
    from src.vectorstore import get_vectorstore
    from src.llm import build_llm, build_answer_prompt, format_output
except Exception:
    from embeddings import build_embeddings
    from vectorstore import get_vectorstore
    from llm import build_llm, build_answer_prompt, format_output


def load_pdf(pdf_path: str | Path) -> List[Document]:
    loader = PyPDFLoader(str(pdf_path))
    return loader.load()


def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 150) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)


def ingest_pdf_into_qdrant(pdf_path: str, collection: str | None = None) -> Dict[str, Any]:
    docs = load_pdf(pdf_path)
    chunks = split_documents(docs)
    embeddings = build_embeddings()
    vs = get_vectorstore(embeddings, collection)
    ids = vs.add_documents(chunks)
    return {"num_chunks": len(chunks), "collection": vs.collection_name, "ids": ids}


def get_retriever(collection: str | None = None, search_k: int = 4) -> VectorStoreRetriever:
    embeddings = build_embeddings()
    vs = get_vectorstore(embeddings, collection)
    return vs.as_retriever(search_kwargs={"k": search_k})


def rag_answer(question: str, collection: str | None = None) -> Dict[str, Any]:
    retriever = get_retriever(collection)
    context_docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([d.page_content for d in context_docs])
    llm = build_llm()
    prompt = build_answer_prompt("You answer questions based on provided PDF context and cite short quotes.")
    chain = prompt | llm
    generated = chain.invoke({"context": context, "question": question})
    answer = format_output(generated)
    return {"answer": answer, "sources": [d.metadata for d in context_docs]}



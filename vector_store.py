import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

FAISS_PATH = "faiss"

def get_vectorstore(chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if os.path.exists(FAISS_PATH):

        vectorstore = FAISS.load_local(
            FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    else:

        vectorstore = FAISS.from_documents(
            chunks,
            embeddings
        )

        vectorstore.save_local(FAISS_PATH)

    return vectorstore
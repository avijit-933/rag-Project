from pdf_loader import load_and_split_pdf
from vector_store import get_vectorstore
from rag_chain import stream_answer

PDF_PATH = "data/My.pdf"

print("Loading PDF...")

chunks = load_and_split_pdf(PDF_PATH)

print(f"Chunks Created: {len(chunks)}")

vectorstore = get_vectorstore(chunks)

print("RAG System Ready!")
print("Type 'exit' to quit.")

while True:

    query = input("\nAsk Question: ")

    if query.lower() == "exit":
        break

    answer = stream_answer(
        vectorstore,
        query
    )

    print("\nAnswer:")
    print(answer)
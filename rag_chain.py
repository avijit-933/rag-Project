

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)
def stream_answer(vectorstore, query):

    docs = vectorstore.similarity_search(
        query,
        k=3
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are a helpful assistant.

Use ONLY the context below.
If the answer is not found in the context, say:
"I couldn't find that information in the uploaded document."

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    for chunk in llm.stream(prompt):
        if hasattr(chunk, "content"):
            yield chunk.content
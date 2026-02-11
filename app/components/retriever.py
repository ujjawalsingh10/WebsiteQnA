# from langchain_classic.chains import Re
from langchain_core.prompts import PromptTemplate
from app.components.vector_store import load_vector_store
from app.components.llm import load_llm
from app.config.config import HUGGINGFACE_REPO_ID, HF_TOKEN
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

prompt = PromptTemplate(
    template="""
    You are a technical assistant helping analyze website content.

    Use ONLY the provided context.
    You may infer the website name from titles, headers, or visible branding.

    If the answer cannot be determined from the context, say:
    "I don't have enough information."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """,
        input_variables=["context", "question"]
)

parser = StrOutputParser()

db = load_vector_store()

llm = load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN)

retriever = db.as_retriever(search_type = 'similarity', search_kwargs={'k' : 4})

# rag_chain = (
#     {
#         'context' : retriever,
#         'question' : RunnablePassthrough()
#     }
#     | prompt
#     | llm
#     | parser
# )

# print("\n Rage system ready. Type 'exit' to quit \n")

# while True:
#     query = input('Ask a question: ')

#     if query.lower() == 'exit':
#         break

#     response = rag_chain.invoke(query)

#     print('\nAnswer:\n')
#     print(response.content if hasattr(response, "content") else response)
#     print("\n" + "-" * 50 + "\n")

while True:
    query = input('Ask a question: ')

    if query.lower() == 'exit':
        break

    # STEP 1: Retrieve documents manually
    retrieved_docs = retriever.invoke(query)

    # STEP 2: Write retrieved content to debug file
    with open("retrieved_debug.txt", "w", encoding="utf-8") as f:
        f.write(f"Query: {query}\n\n")
        for i, doc in enumerate(retrieved_docs):
            f.write(f"\n--- Document {i+1} ---\n")
            f.write(doc.page_content)
            f.write("\n\n")

    # STEP 3: Combine context manually
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # STEP 4: Call prompt → LLM → parser manually
    formatted_prompt = prompt.invoke({
        "context": context_text,
        "question": query
    })

    response = llm.invoke(formatted_prompt)
    final_answer = parser.invoke(response)

    print("\nAnswer:\n")
    print(final_answer)
    print("\n" + "-" * 50 + "\n")
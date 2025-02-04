import os
from langchain_core.prompts import PromptTemplate

from pathlib import Path
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

messages = []

class ResponseSchema(BaseModel):
    text: str = Field(description="Answer to the query")


def rag_prompt():
    prompt = """
    User has a monthly saving of {savings} INR. Recommend the best SIP (Systematic Investment Plan) from the following list:
        {investments}
        Provide a detailed explanation of why this SIP is the most suitable option for the user’s savings goal.

        output -
        {{
            "text": "give response here"
        }}

    """
    return prompt

def prompt_chain(prompt,schema):
    """
    Create and return a prompt chain for generating responses.

    This function sets up a PromptTemplate with specific instructions for the AI assistant,
    initializes a ChatGoogleGenerativeAI model, and combines them into a chain that can
    generate responses based on input queries and contexts.

    Returns:
        chain: A callable chain that takes a query and context as input and returns a generated response.
    """
    # Set up the parser with the ResponseSchema structure
    parser = JsonOutputParser(pydantic_object=schema)

    # Define the prompt template
    prompt = PromptTemplate.from_template(prompt)

    llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=1.0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
        )


    chain = prompt | llm | parser

    return chain

def model_response(query: str) -> dict:
    """
    Process and generate model response based on query.

    Args:
        query (str): User query string

    Returns:
        dict: Response containing data, text and similar documents
    """
    try:

        retrieved_data,similar_models_doc = retrieve_data(query)
        # Update message history
        messages.append({
            "role": "user",
            "content": query
        })
        chain = prompt_chain(rag_prompt(), ResponseSchema)
        response = chain.invoke({
                "savings": savings,
                "investments": retrieved_data
            })

        return {
            "data": response,
            "similar":similar_models_doc,
        }

    except Exception as e:
        print(f"Error in model_response: {str(e)}")
        raise

def retrieve_data(query: str):
    """
    Retrieves and processes data, prioritizing "product_name" documents.
    """

    retriever, embedding = get_data()
    if retriever is None or embedding is None:
        print("Vector retriever or embedding model not initialized.")
        raise RuntimeError("Vector retriever or embedding model not initialized.")

    try:
        most_similar_docs = retriever.vectorstore.similarity_search(query, k=35)
        doc_text = ''
        for doc in most_similar_docs:
            doc_text = "\n".join(doc.page_content)
        return doc_text,most_similar_docs

    except Exception as e:
        print(f"Error processing query: {e}")
        raise

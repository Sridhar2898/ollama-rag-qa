"""Import the required libraries and modules"""
import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate

st.sidebar.title('RAG App') # Set the title of the streamlit app
QUERY = None

# Set the Ollama parameters
OLLAMA_URL="http://ollama:11434"
LLM_MODEL="llama3"
EMBEDDING_MODEL="all-minilm"

# Define LLM and EMBEDDING
# pylint: disable=not-callable
llm = Ollama(
    base_url=OLLAMA_URL,
    model=LLM_MODEL
)

embeddings = OllamaEmbeddings(
    base_url=OLLAMA_URL,
    model=EMBEDDING_MODEL
)

def create_vector(text):
    """Create the vector from the text"""
    vector_content = FAISS.from_documents(text, embeddings)
    return vector_content

def generate_prompt():
    """Generate the prompt"""
    prompt = """
      1. Use the following pieces of context to answer the question at the end.
      2. If you don't know the answer, just say that "I don't know" but don't make up an answer on your own.\n
      3. Keep the answer crisp and limited to 3,4 sentences.

      Context: {context}

      Question: {question}

      Helpful Answer:"""
    return prompt

# upload a PDF file using streamlit
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Create a temporary file and write the uploaded file's bytes to it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        temp_path = tmp.name
    # Create text content from the pdf file
    loader = PyPDFLoader(temp_path)
    docs = loader.load()
    os.unlink(temp_path)

    # Create a vector from the text content
    with st.spinner('Creating the vector...'):
        vector = create_vector(docs)
        st.sidebar.write("Vector created successfully")

    # Ask the user to enter the query
    query = st.text_input("Enter the query")

    # Run the query if the button is pressed.
    if st.button('Submit Query', type='primary'):
        # Define the document retriever
        retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        # Retrieve documents from the vector based on the query
        retrieved_docs = retriever.invoke(query)

        # Generate a prompt for the QA chain
        PROMPT = generate_prompt()
        QA_CHAIN_PROMPT = PromptTemplate.from_template(PROMPT)

        # Define the QA chain
        llm_chain = LLMChain(
          llm=llm,
          prompt=QA_CHAIN_PROMPT,
          callbacks=None,
          verbose=True
        )

        # Define the document prompt
        document_prompt = PromptTemplate(
          input_variables=["page_content", "source"],
          template="Context:\ncontent:{page_content}\nsource:{source}",
        )

        # Define the combine documents chain
        combine_documents_chain = StuffDocumentsChain(
          llm_chain=llm_chain,
          document_variable_name="context",
          document_prompt=document_prompt,
          callbacks=None,
        )

        # Define the retrieval QA chain
        qa = RetrievalQA(
          combine_documents_chain=combine_documents_chain,
          retriever=retriever,
        )

        # Invoke the query
        with st.spinner('Generating the answer...'):
            answer = qa.invoke(query)

        # Display the answer
        st.write(answer["result"])

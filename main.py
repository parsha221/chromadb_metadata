import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
import gradio as gr
 
# Your OpenAI API key
OPENAI_API_KEY = ""
 
def process_files(files):
    text_chunks = []
    chunk_sources = {}
   
    for file in files:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
       
        # Break text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            separators="\n",
            chunk_size=30000,
            chunk_overlap=100,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        text_chunks.extend(chunks)
        for chunk in chunks:
            chunk_sources[chunk] = file
   
    # Generate embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
   
    # Create vector store with Chroma
    vector_store = Chroma.from_texts(text_chunks, embeddings)
    return vector_store, chunk_sources
 
def answer_question(vector_store, chunk_sources, user_question):
    # Perform similarity search
    match = vector_store.similarity_search(user_question)
   
    # Define the language model
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        temperature=0,
        max_tokens=4000,
        model_name="gpt-4-turbo"
    )
   
    # Load QA chain and get response
    chain = load_qa_chain(llm, chain_type="stuff")
    response = chain.run(input_documents=match, question=user_question)
   
    # Determine which file the response came from
    matched_file = chunk_sources[match[0].page_content]
   
    return response, matched_file
 
def gradio_interface(file1, file2, file3, user_question):
    files = [file1, file2, file3]
    files = [file for file in files if file is not None]
   
    if len(files) > 0:
        vector_store, chunk_sources = process_files(files)
        if user_question:
            response, matched_file = answer_question(vector_store, chunk_sources, user_question)
            return {
                "response": response,
                "matched_file": matched_file
            }
    return "Please upload one or more files and enter a question."
 
# Create Gradio interface
iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.components.File(type="filepath", label="Upload PDF file 1"),
        gr.components.File(type="filepath", label="Upload PDF file 2"),
        gr.components.File(type="filepath", label="Upload PDF file 3"),
        gr.components.Textbox(label="Type Your question here")
    ],
    outputs="json",
    title="My First Chatbot PDF",
    description="Upload one or more PDF files and start asking questions."
)
 
iface.launch()

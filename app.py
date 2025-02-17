import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
import tempfile
import os
import pypdf
from langchain.chains import RetrievalQA


# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.cache_resource.clear()

# Streamlit UI
st.title("📄 AI-Powered &A")
st.write("Upload a PDF and ask questions based on its content.")

# Initialize variables in session state
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

# File Upload
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name

        # First, verify the PDF is readable
        try:
            pdf_reader = pypdf.PdfReader(pdf_path)
            if len(pdf_reader.pages) == 0:
                st.error("The uploaded PDF appears to be empty. Please check the file and try again.")
                os.unlink(pdf_path)
                st.stop()
                
            # Display number of pages found
            st.info(f"PDF loaded successfully. Number of pages: {len(pdf_reader.pages)}")
            
            if pdf_reader.is_encrypted:
                st.error("The PDF is encrypted/password protected. Please remove the password and try again.")
                os.unlink(pdf_path)
                st.stop()
                
        except pypdf.errors.PdfReadError as e:
            st.error(f"Error reading PDF: The file appears to be corrupted or invalid. Error: {str(e)}")
            os.unlink(pdf_path)
            st.stop()

        # Load PDF with LangChain
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        if not documents:
            st.error("No text could be extracted from the PDF. Please ensure the PDF contains readable text.")
            os.unlink(pdf_path)
            st.stop()

        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        
        if not chunks:
            st.error("No useful text chunks could be extracted from the PDF.")
            os.unlink(pdf_path)
            st.stop()

        # Create embeddings and vector store
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever()

        template = """Based on the following context, provide a direct and concise answer to the question. Do not include any reasoning tags or metadata in your response. If you don't know the answer, simply say "I don't know."

Context: {context}
Question: {question}

Your answer should be straightforward and to the point. Do not include any <think> tags or similar markup in your response.

Answer:"""
        QUESTION_PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "question"])


        # Initialize LLM
        llm = ChatGroq(
            model_name="deepseek-r1-distill-qwen-32b",
            api_key="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"
        )

        # Create QA chain
        st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
)

        st.success("PDF processed successfully! You can now ask questions.")

        # Clean up temporary file
        os.unlink(pdf_path)

    except Exception as e:
        st.error(f"Detailed error information: {str(e)}")
        st.error("If you're seeing this error, please ensure your PDF:")
        st.write("1. Is not empty")
        st.write("2. Is not corrupted")
        st.write("3. Is not password protected")
        st.write("4. Contains readable text (not just scanned images)")
        if 'pdf_path' in locals():
            os.unlink(pdf_path)

# Question input and response
if st.session_state.qa_chain is not None:
    user_question = st.text_input("Ask a question about the document:")
    
    if user_question:
        try:
            # Get the response
            response = st.session_state.qa_chain(
            {"question": user_question, "chat_history": st.session_state.chat_history}
            )
            
            # Display just the answer without any additional context
            st.write(response['answer'].strip())
            
            # Update chat history
            st.session_state.chat_history.append(
                (user_question, response['answer'])
            )
            
        except Exception as e:
            st.error(f"Error getting response: {str(e)}")
else:
    st.info("Please upload a PDF first to start asking questions.")
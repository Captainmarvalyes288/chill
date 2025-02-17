# AI-Powered Document QA

## 📌 Overview
This is a **Streamlit-based** AI-powered document question-answering application that allows users to upload a **PDF** and ask questions based on its content. The system uses **MapReduce** to process the document and generate answers based on all extracted text.

## 🚀 Features
- **PDF Upload & Processing**: Extracts text from PDFs and splits them into meaningful chunks.
- **MapReduce-based QA**: Answers questions by processing all document chunks before aggregating a final response.
- **Handles Encrypted PDFs**: Ensures that password-protected PDFs are not processed.
- **Automatic Text Cleanup**: Removes unwanted `<think>` tags from responses.
- **Chat History**: Maintains a session-based chat history.

## 🛠️ Tech Stack
- **Python** (Backend Logic)
- **Streamlit** (Frontend UI)
- **LangChain** (LLM and NLP processing)
- **PyMuPDF & PyPDF** (PDF Processing)
- **FAISS** (Vector Storage )
- **HuggingFace Embeddings** (For text chunking & understanding)
- **Groq Chat Model** (For answering questions)

## 📂 Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/rohannso/ai_powered_pdf_qanda
cd ai_powered_pdf_qanda
```

### 2️⃣ Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate     # On Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Application
```bash
streamlit run app.py
```

## 📜 How It Works
1. **Upload a PDF**: The app extracts and processes text into chunks.
2. **Ask a Question**: The AI processes all text chunks and applies the **MapReduce** method to generate an accurate response.
3. **Receive Answer**: The AI aggregates responses from multiple document sections before providing the final answer.

## 🛠 Configuration
- **LLM Model**: The application uses `ChatGroq` with `deepseek-r1-distill-qwen-32b`. If needed, update the model in `app.py`.
- **Chunk Size & Overlap**: Adjusted in `CharacterTextSplitter` for better accuracy.

## 💡 Future Enhancements
- 🔹 Add **FAISS similarity search** as an option for faster querying.
- 🔹 Enable support for multiple document formats (**Word, TXT**).
- 🔹 Provide **PDF summarization** along with QA.



---
📩 **For Issues & Contributions:** Open a pull request or report issues in the repository.


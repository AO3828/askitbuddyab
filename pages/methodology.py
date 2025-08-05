import streamlit as st

# Title of the page
st.title("Methodology")

# Section 1: Overview
st.header("How this App Works")
st.markdown("""
This chatbot prototype uses a Retrieval-Augmented Generation (RAG) approach to answer onboarding-related questions using internal documents.

It combines document retrieval with OpenAI's GPT-4o model to provide relevant and contextual answers, based only on the uploaded onboarding packs.
""")

# Section 2: Architecture
st.header("Technical Methodology")
st.markdown("""
The app works in the following steps:

1. **Document Upload (Admin)**  
   - Admins upload PDF onboarding packs (e.g., Day 1, Week 1, Month 1).

2. **Document Parsing and Chunking**  
   - PDFs are loaded and split into smaller text chunks using `RecursiveCharacterTextSplitter`.
   - Chunk size is optimized to preserve topic context.

3. **Embeddings & Indexing**  
   - Each text chunk is converted into a vector using OpenAIEmbeddings.
   - Vectors are stored in a FAISS vector store for fast similarity search.

4. **Query Handling (User)**  
   - Users ask questions using natural language.
   - The app finds the most relevant chunks using vector similarity.
   - The selected chunks are passed to GPT-4o, which generates a concise, helpful answer based only on the document content.

5. **Answering Policy**  
   - If no relevant content is found, the bot will say:  
     "I don't know. Thanks for asking!"
""")

# Section 3: Tools & Libraries
with st.expander("Libraries and Tools Used"):
    st.write("""
- Streamlit: UI framework
- LangChain: RAG pipeline and document processing
- OpenAI GPT-4o: Language model for answering
- FAISS: Vector similarity search
""")

# Section 4: Future Enhancements
st.header("Possible Enhancements")
st.markdown("""
- Add answer export (PDF or text)
- Support for summarization
- Simple login system for role-based access
- Session history for user queries
""")

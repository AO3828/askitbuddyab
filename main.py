import os
import streamlit as st
from streamlit_option_menu import option_menu
from dotenv import load_dotenv
from tempfile import NamedTemporaryFile
from pathlib import Path
from helper_functions import utility
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
#from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from difflib import get_close_matches

#Set up OpenAI key
#if load_dotenv('.env'):
OPENAI_KEY = os.getenv('OPENAI_API_KEY')
#else:
#    OPENAI_KEY = st.secrets['OPENAI_API_KEY']

# Prompt template
def get_prompt_template():
    return PromptTemplate.from_template("""
Use the following context to answer the question at the end. 
If you don't have a clear answer, answer based on the best available clues from the context. 
If there's absolutely no information, say you don't know. 
Be concise and end with 'Thanks for asking!'.

{context}
Question: {question}
Helpful Answer:
""")

# --- FAQ MAP FOR LLM REWRITES --- 
FAQ_MAP = { 
    # ITD/CIO Related 
    "who is cio": "Who is the Chief Information Officer (CIO) of Singapore Customs?", 
    "who is customs cio": "Who is the Chief Information Officer (CIO) of Singapore Customs?", 
    "who is custom cio": "Who is the Chief Information Officer (CIO) of Singapore Customs?", 
    "who is head of it": "Who is the Chief Information Officer (CIO) of Singapore Customs?", 
    "who is head of info tech": "Who is the Chief Information Officer (CIO) of Singapore Customs?", 
    "who is the boss of itd": "Who is the Chief Information Officer (CIO) of Singapore Customs?", 
    "who manages itd": "Who is the Chief Information Officer (CIO) of Singapore Customs?", 
    "who manages the it division": "Who is the Chief Information Officer (CIO) of Singapore Customs?", 
    "who is in charge of the it department": "Who is the Chief Information Officer (CIO) of Singapore Customs?", 
    "who is cheryl": "Who is Cheryl in Singapore Customs?", 
    "who is the itd boss": "Who is the Chief Information Officer (CIO) of Singapore Customs?", 
    "who is the information technology director": "Who is the Chief Information Officer (CIO) of Singapore Customs?", 
    "who is in charge of ITD": "Who is the Chief Information Officer (CIO) of Singapore Customs?", 
    "who is head of ITD": "Who is the Chief Information Officer (CIO) of Singapore Customs?", 
    "who is the head of IT directorate": "Who is the Chief Information Officer (CIO) of Singapore Customs?", 
    "who leads the IT division": "Who is the Chief Information Officer (CIO) of Singapore Customs?", 

    # Customs Leadership 
    "who is the head of customs": "Who is the Director-General (DG) of Singapore Customs?", 
    "who leads customs": "Who is the Director-General (DG) of Singapore Customs?", 
    "who is the dg of customs": "Who is the Director-General (DG) of Singapore Customs?", 
    "who is customs boss": "Who is the Director-General (DG) of Singapore Customs?", 

    # GovTech Leadership 
    "who is the head of govtech": "Who is the Chief Executive of GovTech Singapore?", 
    "who leads govtech": "Who is the Chief Executive of GovTech Singapore?", 
    "who is govtech boss": "Who is the Chief Executive of GovTech Singapore?"
} 


def rewrite_query_if_needed(query: str) -> str: 
    lower_q = query.strip().lower() 
    matches = get_close_matches(lower_q, FAQ_MAP.keys(), n=1, cutoff=0.7) 
    if matches: 
        rewritten = FAQ_MAP[matches[0]] 
        st.caption(f"Rewritten Query: {rewritten}") 
        return rewritten 
    return query 

# Build retriever from PDFs in the data folder
def build_retriever_from_data_folder(data_folder="data"):
    all_docs = []
    data_path = Path(data_folder)
    data_path.mkdir(exist_ok=True)

    for file in data_path.glob("*.pdf"):
        loader = PyPDFLoader(str(file))
        all_docs.extend(loader.load())

    if not all_docs:
        st.error("No PDF documents found in the data folder.")
        return None

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    child_docs = child_splitter.split_documents(all_docs)

    embeddings = OpenAIEmbeddings(base_url="https://llmaas.govtext.gov.sg/gateway")
    vectorstore = FAISS.from_documents(child_docs, embedding=embeddings)
    docstore = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        parent_splitter=parent_splitter,
        child_splitter=child_splitter
    )
    retriever.add_documents(all_docs)
    return retriever

# Display answer and chunks
def handle_user_query(qa_chain, user_query):
    result = qa_chain(user_query)
    st.write("### Answer:")
    st.write(result["result"])
#    with st.expander("Show Retrieved Chunks"):
#        for i, doc in enumerate(result["source_documents"]):
#            st.markdown(f"**Chunk {i+1}:**")
#            st.code(doc.page_content[:1000])

# Customized onboarding suggestions per week
def show_onboarding_guidance(week, qa_chain=None):
    if week == "Day 1+":
        st.write("**Welcome to Day 1-7 of your onboarding!**")
        questions = [
            "What is the IT Division's structure and key contacts?",
            "How do I install required software or get IT help?",
            "Where can I find the Customs Intranet and IT Portal?",
            "What are the important websites I should note?",
            "How is data classified and why does it matter?",
            "How do I connect to SilverGate Wifi and access shared data?"
        ]
    elif week == "Week 1+":
        st.write("**Welcome to Week 2 to 4 of your onboarding!**")
        questions = [
            "What are the main IT systems used by Customs and how are they classified?",
            "Where can I find the Customs IT Security Policy?",
            "How do I manage procurement or submit IT budget requests?",
            "What is ERMS and how do I handle official email records?",
            "How do I book meeting rooms?"
        ]
    elif week == "Week 4+":
        st.write("**Welcome to Week 4+ of your onboarding!**")
        questions = [
            "What are recurring tasks?",
            "How do I submit expense claims or finance documents correctly?",
            "What's the proper format for writing meeting minutes?",
            "How do I check my payslip or leave balance online?",
            "What is IM8 and why is it important for government ICT compliance?",
            "Where do I find training programs in Workday or Learn, and how do I apply?"
        ]

    if qa_chain:
        st.markdown("**Need help with something specific? Click a question below:**")
        for question in questions:
            if st.button(question):
                with st.spinner(f"Fetching the answer for: {question}"):
                    handle_user_query(qa_chain, question)


# Simple login screen
#def login():
#    st.title("🔐 Login to AskITBuddy")
#    with st.form("login_form"):
#        username = st.text_input("Username")
#        password = st.text_input("Password", type="password")
#        submit = st.form_submit_button("Login")

#        if submit:
#            if username in CREDENTIALS and CREDENTIALS[username] == password:
#                st.session_state["authenticated"] = True
#                st.session_state["username"] = username
#                st.success(f"Welcome, {username}!")
#                st.rerun()
# Do not continue if check_password is not True.  

#if not utility.check_password():  
#    st.stop()
#            else:
#                st.error("Invalid username or password.")


# Style sheet for text of input text box
tabs_font_css = """
<style>
div[class*="stTextArea"] label p {
  font-size: 24px;
  color: red;
}

div[class*="stTextInput"] label p {
  font-size: 24px;
}

div[class*="stSelect"] label p {
  font-size: 24px;
}
</style>
"""

st.write(tabs_font_css, unsafe_allow_html=True)

#st.text_area("Text area")
#st.text_input("Text input")
#st.number_input("Number input")

#########


# App main
def main():
    st.set_page_config(page_title="AskITBuddy", layout="centered")

    #if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
    #    login()
    #    return

    if not utility.check_password():  
        st.stop()

    with st.sidebar:
        st.session_state["username"] = st.selectbox(
            "Logged in as:",
            options=["user", "admin"])

    #st.session_state["username"] = option_menu(
    #menu_title="Please select your role:",  # Required
    #options=["user", "admin"],  # Required
    #icons=["house","gear"], menu_icon="tencent-qq", default_index=0)  # Optional (Bootstrap icons)

    #with st.sidebar:
    #    selected = option_menu("Main Menu", ["Home", 'Settings'], 
    #        icons=['list-task', 'gear'], menu_icon="cast", default_index=0)
    #    selected
    
    username = st.session_state["username"]

    #st.sidebar.success(f"Logged in as: {username}")

    # Admin UI
    if username == "admin":
        st.header("📁 Admin Dashboard")

        st.write("")
        filelist=[]
        for root, dirs, files in os.walk("data"):
            for file in files:
                filename=os.path.join(root, file)
                st.markdown(filename)
                filelist.append(filename)

        #with st.expander("Click to see the uploaded files"):
        #    st.write(filelist)

        st.write("")
        st.subheader("Delete file")
        file_to_delete = st.text_input("Enter the <path>/<filename> to delete:")
        if st.button("Delete File"):
            utility.delete_file(file_to_delete)

        st.write("")
        st.subheader("Upload files:")
        uploaded_files = st.file_uploader("Upload PDF files to the data folder", type=["pdf"], accept_multiple_files=True)
        if uploaded_files:
            data_path = Path("data")
            data_path.mkdir(exist_ok=True)
            for file in uploaded_files:
                save_path = data_path / file.name
                with open(save_path, "wb") as f:
                    f.write(file.getbuffer())
            st.success("Files uploaded successfully. Please refresh or rerun to build retriever.")
        return

    # User UI
    st.header("AskITBuddy - Your IT Onboarding Assistant")

    retriever = build_retriever_from_data_folder("data")
    if not retriever:
        st.warning("No retrievable documents found. Ask the Admin to upload PDFs first.")
        return

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(base_url="https://llmaas.govtext.gov.sg/gateway", model="gpt-4o", temperature=0),
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": get_prompt_template()}
    )

    onboarding_week = st.selectbox(
        "Which week of your onboarding are you in?",
        options=["Day 1+", "Week 1+", "Week 4+"]
    )
    show_onboarding_guidance(onboarding_week, qa_chain=qa_chain)

    user_query = st.text_input("Ask me anything about your IT onboarding:")
    if user_query:
        with st.spinner("Thinking..."):
            handle_user_query(qa_chain, user_query)

# Run it
if __name__ == "__main__":
    main()

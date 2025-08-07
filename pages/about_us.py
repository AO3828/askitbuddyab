# about_us.py
import streamlit as st

# Title of the page
st.title("About this App")

# Section 1: Company Info
st.header("About this App")
st.write("""
    This is a chatbot POC that uses the OpenAI API to generate text completions. 
    The aim is to enable new IT joiners in Customs to get familiar with the IT Directorate and Customs as an organization, and streamlining the onboarding experience by:
    - Answering questions like "Where do I find information on security policies?"; "Where is the org chart?"; "Where do I find information on a particular system in Customs?".
    - Responding to natural language questions from new joiners.
    - Providing contextual, accurate and conversational answers.
""")

# Section 2: How to use this app
with st.expander("How to use this App"):
    st.write("1. Enter your prompt in the text area or click on the predefined prompts.")
    st.write("2. The app will generate a text completion based on your prompt.")

# Section 3: Contact Info or Additional Details (Optional)
st.header("Contact Us")
st.write("""
    If you have any questions or need assistance, feel free to reach out:
    - **Email**: support@example.com
    - **Phone**: (65) 0000 0000
""")

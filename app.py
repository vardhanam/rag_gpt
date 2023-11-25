import streamlit as st
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os

#function to fetch text data from the links
def fetch_article_content(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        st.error(f"Error fetching {url}: {e}")
        return ""

#function to collate all the text into a string
def process_links(links):
    all_contents = ""
    for i, link in enumerate(links):
        content = fetch_article_content(link.strip())
        all_contents += content + "\n\n"
    return all_contents

#function to chunk the articles
def get_text_chunks_langchain(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(text)
    return texts

def main():
    st.title('News Article Fetcher')

    # Initialize state variables
    if 'articles_fetched' not in st.session_state:
        st.session_state.articles_fetched = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = ""


    # Model selection
    model_choice = st.radio("Choose your model", ["GPT 3.5", "GPT 4"], key= "model_choice")
    model = "gpt-3.5-turbo-1106" if st.session_state.model_choice == "GPT 3.5" else "gpt-4-1106-preview"

    #API_KEY
    API_KEY = st.text_input("Enter your OpenAI API key", type="password", key= "API_KEY")

    # Ensure API_KEY is set before proceeding
    if not API_KEY:
        st.warning("Please enter your OpenAI API key.")
        st.stop()



    uploaded_file = st.file_uploader("Upload a file with links", type="txt")


    # Read the file into a list of links
    if uploaded_file:
        stringio = uploaded_file.getvalue().decode("utf-8")
        links = stringio.splitlines()

    # Fetch the articles' content
    if st.button("Fetch Articles") and uploaded_file:
        progress_bar = st.progress(0)
        with st.spinner('Fetching articles...'):
            article_contents = process_links(links)
            progress_bar.progress(0.25)  # Update progress to 25%

            #Process the article contents
            texts = get_text_chunks_langchain(article_contents)
            progress_bar.progress(0.5)  # Update progress to 50%

            #storing the chunked articles as embeddings in Qdrant
            os.environ["OPENAI_API_KEY"] =  st.session_state.API_KEY
            embeddings = OpenAIEmbeddings()
            vector_store = Qdrant.from_texts(texts, embeddings, location=":memory:",)
            retriever = vector_store.as_retriever()
            progress_bar.progress(0.75)  # Update progress to 75%

            #Creating a QA chain against the vectorstore
            llm = ChatOpenAI(model_name= model)
            if 'qa' not in st.session_state:
                st.session_state.qa = RetrievalQA.from_llm(llm= llm, retriever= retriever)
            progress_bar.progress(1)

            st.success('Articles fetched successfully!')
            st.session_state.articles_fetched = True

    if 'articles_fetched' in st.session_state and st.session_state.articles_fetched:

        # Chatbot-like interface
        # Initialize a variable to store chat history
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = ""


        query = st.text_input("Enter your query here:", key="query")

        if query:
            # Process the query using your QA model (assuming it's already set up)
            with st.spinner('Analyzing query...'):
                qa = st.session_state.qa
                response = qa.run(st.session_state.query)  # Replace with your actual call to the QA model
            # Update chat history
            st.session_state.chat_history += f"> {st.session_state.query}\n{response}\n\n"



        # Display conversation history
        st.text_area("Conversation:", st.session_state.chat_history, height=1000, key="conversation_area")
                # JavaScript to scroll to the bottom of the text area
        st.markdown(
            f"<script>document.getElementById('conversation_area').scrollTop = document.getElementById('conversation_area').scrollHeight;</script>",
            unsafe_allow_html=True
        )


if __name__ == "__main__":
    main()







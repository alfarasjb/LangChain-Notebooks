# Import langchain dependencies 
from langchain.document_loaders import PyPDFLoader 
from langchain.indexes import VectorstoreIndexCreator 
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_openai import ChatOpenAI 
# Env 
from dotenv import load_dotenv 
import os 
# Streamlit for UI 
import streamlit as st

load_dotenv()
# Setup llm 
llm = ChatOpenAI(model_name="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))


# Custom Data
@st.cache_resource
def load_pdf():
    filename = 'data/monopoly.pdf'
    loaders = [PyPDFLoader(filename)]
    index = VectorstoreIndexCreator(
        embedding=OpenAIEmbeddings(),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    ).from_loaders(loaders)
    return index


index = load_pdf()

# Create QA Chain
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=index.vectorstore.as_retriever(),
    input_key='question'
)

# Setup app title 
st.title("Ask WatsonX")

# Session
if 'messages' not in st.session_state: 
    st.session_state.messages = [] 

for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Prompt 
prompt = st.chat_input('Say something')

if prompt: 
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = chain.run(prompt)
    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
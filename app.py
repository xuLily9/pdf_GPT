import os 
import streamlit as st
from apikey import apikey 
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

# os.environ['HUGGINGFACEHUB_API_TOKEN']= apikey
os.environ['OPENAI_API_KEY']= apikey
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader =PdfReader(pdf)
        for page in pdf_reader.pages: 
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(mempry_ley = 'chat_history', return_message= True)
    conversation_chain= ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain



def main():
    st.set_page_config(page_title="Chat with PDFs")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with PDFs")
    st.text_input("Ask question about your documents")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs=st.file_uploader("Upload your PDFs here and click on process", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
            #get the pdfs
                raw_text = get_pdf_text(pdf_docs)
                # st.write(raw_text)
                text_chunks= get_text_chunks(raw_text)
                # st.write(text_chunks)
                vectorstore = get_vectorstore(text_chunks)
                # st.session_state.conversation = get_conversation_chain(
                #     vectorstore)




if __name__ =='__main__':
    main()
import warnings
import streamlit as st
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough

warnings.filterwarnings('ignore')
_ = load_dotenv()

persistent_dir = './vectorstore'

llm = ChatAnthropic(model="claude-3-5-sonnet-20240620",
                    temperature=0,
                    timeout=None,
                    streaming=True)

vectorstore = Chroma(persist_directory=persistent_dir,
                     embedding_function=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context, which can include text and tables:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# RAG pipeline
chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

st.subheader("Chat with Knowledgebase:")
st.write("nvidia 8-k : What is the basic net income per share for the three months ended April 28, 2024 for Nvidia?")
st.write("apple 8-k: What’s the iPhone Net sales for Three Months Ended March 30, 2024 in millions of $ for Apple?")

request = st.text_area('How can I help you today? ', height=100)
submit = st.button("submit", type="primary")

if request and submit:
    response = chain.invoke(request)
    st.write(response)

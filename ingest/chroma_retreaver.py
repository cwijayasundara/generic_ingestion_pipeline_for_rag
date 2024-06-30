import warnings
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings

warnings.filterwarnings('ignore')
_ = load_dotenv()

persistent_dir = '../vectorstore'

vectorstore = Chroma(persist_directory=persistent_dir,
                     embedding_function=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()

llm = ChatAnthropic(model="claude-3-5-sonnet-20240620",
                    temperature=0,
                    timeout=None,
                    streaming=True)


def chroma_db_upload_verifier(query):

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

    return chain.invoke(query)

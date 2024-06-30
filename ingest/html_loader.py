import warnings
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from unstructured.partition.html import partition_html
from langchain_core.documents import Document

warnings.filterwarnings('ignore')
_ = load_dotenv()


# Load HTML content from a file

def upload_html_file_to_vector_db(html_file_path, persistent_dir):

    html_elements = partition_html(filename=html_file_path,
                                   chunking_strategy="by_title",
                                   max_characters=4096,
                                   new_after_n_chars=3800,
                                   combine_text_under_n_chars=2000)

    documents = []
    for element in html_elements:
        metadata = element.metadata.to_dict()
        del metadata["languages"]
        metadata["source"] = metadata["filename"]
        documents.append(Document(page_content=element.text, metadata=metadata))

    # Filter out elements with complex metadata that are not useful for the vector store
    documents = filter_complex_metadata(documents)

    Chroma.from_documents(documents,
                          OpenAIEmbeddings(),
                          persist_directory=persistent_dir)

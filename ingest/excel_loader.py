import warnings
from dotenv import load_dotenv
from unstructured.partition.xlsx import partition_xlsx
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

warnings.filterwarnings('ignore')
_ = load_dotenv()


def upload_excel_file_to_vector_db(excel_file_path, persistent_dir):
    excel_elements = partition_xlsx(filename=excel_file_path,
                                    mode="elements",
                                    infer_table_structure=True,
                                    chunking_strategy="by_title",
                                    max_characters=4096,
                                    new_after_n_chars=3800,
                                    combine_text_under_n_chars=2000)

    documents = []
    for element in excel_elements:
        metadata = element.metadata.to_dict()
        metadata["source"] = metadata["filename"]
        documents.append(Document(page_content=element.text, metadata=metadata))

        # Filter out elements with complex metadata that are not useful for the vector store
        documents = filter_complex_metadata(documents)

        Chroma.from_documents(documents,
                              OpenAIEmbeddings(),
                              persist_directory=persistent_dir)

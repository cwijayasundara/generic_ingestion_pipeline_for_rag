import warnings
import os
from dotenv import load_dotenv
from unstructured_client import UnstructuredClient
from unstructured_client.models import shared
from unstructured_client.models.errors import SDKError
from unstructured.staging.base import dict_to_elements
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

warnings.filterwarnings('ignore')

_ = load_dotenv()

DLAI_API_KEY = os.environ.get("DLAI_API_KEY")
DLAI_API_URL = os.environ.get("DLAI_API_URL")

unstructured_client = UnstructuredClient(
    api_key_auth=DLAI_API_KEY,
    server_url=DLAI_API_URL,
)


def upload_pdf_file_to_vector_db(pdf_file_path, persistent_dir):
    with open(pdf_file_path, "rb") as f:
        files = shared.Files(
            content=f.read(),
            file_name=pdf_file_path,
        )

    req = shared.PartitionParameters(
        files=files,
        strategy="hi_res",
        chunking_strategy="by_title",
        hi_res_model_name="yolox",
        skip_infer_table_types=[],
        pdf_infer_table_structure=True,
    )

    try:
        resp = unstructured_client.general.partition(req)
        pdf_elements = dict_to_elements(resp.elements)
    except SDKError as e:
        print(e)

    documents = []
    for element in pdf_elements:
        metadata = element.metadata.to_dict()
        del metadata["languages"]
        metadata["source"] = metadata["filename"]
        documents.append(Document(page_content=element.text, metadata=metadata))

    Chroma.from_documents(documents,
                          OpenAIEmbeddings(),
                          persist_directory=persistent_dir)

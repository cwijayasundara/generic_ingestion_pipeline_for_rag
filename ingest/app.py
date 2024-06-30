from html_loader import upload_html_file_to_vector_db
from pdf_loader import upload_pdf_file_to_vector_db
from chroma_retreaver import chroma_db_upload_verifier

persistent_dir = '../vectorstore'

html_file_path = '../docs/nvidia_8_k_q1_2024.html'
html_uploader_query = "What is the basic net income per share for the three months ended April 28, 2024 for Nvidia?"
# answer = "6.04"

pdf_file_path = '../docs/apple_8_k_q1_2024.pdf'
pdf_uploader_query = "What’s the iPhone Net sales for Three Months Ended March 30, 2024 in millions of $ for Apple?"
# answer = "45,963"


# test the html uploader
upload_html_file_to_vector_db(html_file_path, persistent_dir)
html_uploader_response = chroma_db_upload_verifier(html_uploader_query)
print("html uploader response", html_uploader_response)

# test the pdf uploader
upload_pdf_file_to_vector_db(pdf_file_path, persistent_dir)
pdf_uploader_response = chroma_db_upload_verifier(pdf_uploader_query)
print("pdf uploader response", pdf_uploader_response)



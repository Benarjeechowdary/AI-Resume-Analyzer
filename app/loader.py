from langchain_community.document_loaders import PyPDFLoader,Docx2txtLoader

def load_document(file_path:str):
    if file_path is None:
        print('File path cannot be empty')
    if file_path.endswith('.pdf'):
        loader=PyPDFLoader(file_path=file_path)
    elif file_path.endswith('.docx'):
        loader=Docx2txtLoader(file_path=file_path)
    else:
        print('Invalid File Format')
    
    docs=loader.load()
    return ' '.join([d.page_content for d in docs])
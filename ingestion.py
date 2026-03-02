import chromadb
from extraction import load_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

def ingest_pdf(path="Natural Language Processing.pdf"):
    print(f"Loading: {path}")
    text = load_pdf(path)
    print(f"Extracted {len(text)} characters")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)
    
    client = chromadb.PersistentClient(path="vectordb/chroma_db")
    
    try:
        client.delete_collection("NLP_Book")
    except:
        pass
        
    col = client.create_collection("NLP_Book")
    
    print(f"Storing {len(chunks)} chunks")
    col.add(
        ids=[f"id_{i}" for i in range(len(chunks))], 
        documents=chunks
    )
    print(f"Done: {len(chunks)} chunks ingested")

if __name__ == "__main__":
    ingest_pdf()

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

def create_chunks_from_pdf(data_path, chunk_size, chunk_overlap):

   # Load the documents from the directory
   loader = DirectoryLoader(data_path, loader_cls=PyPDFLoader)

   # Split the documents into chunks
   text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=chunk_size,
      chunk_overlap=chunk_overlap,
      length_function=len,
      is_separator_regex=False,
   )
   docs = loader.load_and_split(text_splitter=text_splitter)
   return docs

data_path = r"D:\RAG Project1\data"
chunk_size = 500
chunk_overlap = 50

docs = create_chunks_from_pdf(data_path, chunk_size, chunk_overlap)
print("docs are created successfully")



embedding_models = ['BAAI/bge-large-en']
embedding = HuggingFaceEmbeddings(model_name=embedding_models[0], cache_folder='./cache')
print('embedding loaded successfully')


def index_documents_and_retrieve(docs, embedding):

    qdrant = Qdrant.from_documents(
        docs,
        embedding,
        location=":memory:",  # Local mode with in-memory storage only
        collection_name="my_documents",
    )

    retriever = qdrant.as_retriever()

    return retriever

retriever = index_documents_and_retrieve(docs, embedding)
print("retriever loaded successfully")



model_id = "llama3:instruct"
# Load the Llama-3 model using the Ollama
llm = ChatOllama(model=model_id)
print("llm loaded successfully")



def build_rag_chain(llm, retriever):  
    template = """
        Answer the question based only on the following context:
        
        {context}
        
        Question: {question}
        """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context","question"]
        )
    
    rag_chain = (
        {"context": retriever,  "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

rag_chain = build_rag_chain(llm, retriever)
print("rag_chain loaded successfully")

rag_chain.invoke('What is the document saying about ?')



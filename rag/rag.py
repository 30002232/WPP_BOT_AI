#Importações de bibliotecas necessárias
import os
from decouple import config #Para carregar variáveis de ambiente de forma segura

#Importações da LangChain
from langchain.text_splitter import RecursiveCharacterTextSplitter #Divide documentos em blocos de texto
from langchain_community.vectorstores import Chroma #Armazenamento vetorial
from langchain_community.document_loaders import PyPDFLoader #Carrega ficheiros PDF como documentos
from langchain_huggingface import HuggingFaceEmbeddings #Gera embeddings com modelos HuggingFace
from langchain_chroma import Chroma

# Define as variáveis de ambiente para as chaves da API
os.environ['GROQ_API_KEY'] = config('GROQ_API_KEY') #Chave para o modelo da Groq
os.environ['HUGGINGFACE_API_KEY'] = config('HUGGINGFACE_API_KEY') #Chave para usar modelos da HuggingFace

#Execução principal (proteção comum para scripts Python)
if __name__ == '__main__':
    #Caminho do ficheiro PDF com os dados.
    file_path = '/app/rag/data/my_document.pdf'
    
    # Carrega o PDF e extrai o texto como uma lista de documentos
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    #Divide os documentos em trechos para facilitar o processamento
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, #Cada bloco terá até 1000 caracteres
        chunk_overlap=200, #Sobreposição entre blocos (para contexto contínuo)
    )
    # Aplica a divisão
    chunks = text_splitter.split_documents(
        documents=docs,
    )

    #Diretório onde os dados vetoriais serão guardados
    persist_directory = '/app/chroma_data'

    #Cria o modelo de embeddings (usa por padrão 'all-mpnet-base-v2')
    embedding = HuggingFaceEmbeddings()

    #Inicializa o armazenamento vetorial Chroma
    vector_store = Chroma(
        embedding_function=embedding,
        persist_directory=persist_directory,
    )
    
    # Adiciona os pedaços de texto ao armazenamento vetorial (para posterior recuperação)
    vector_store.add_documents(
        documents=chunks,
    )

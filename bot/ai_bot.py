#Importações de bibliotecas necessárias
import os

from decouple import config #Permite carregar variáveis de ambiente de forma segura

#Importações da LangChain e integrações relacionadas
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma #Armazenamento vetorial
from langchain_core.messages import HumanMessage, AIMessage #Modelos de mensagens
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder #Templates de prompt para o modelo
from langchain_groq import ChatGroq #Integração com o modelo da Groq
from langchain_huggingface import HuggingFaceEmbeddings #Para gerar embeddings com modelos do Hugging Face
from langchain_chroma import Chroma

#Define a variável de ambiente com a chave da API Groq
os.environ['GROQ_API_KEY'] = config('GROQ_API_KEY') #Carrega do ficheiro .env

#Classe principal do chatbot com inteligência artificial
class AIBot:

    def __init__(self):
        #Inicializa o modelo de chat da Groq (LLaMA 3)
        self.__chat = ChatGroq(model='llama-3.3-70b-versatile')
        #Constrói e guarda o "retriever", que vai buscar documentos relevantes com base na pergunta
        self.__retriever = self.__build_retriever()

    def __build_retriever(self):
        #Define o caminho onde os dados vetoriais estão guardados
        persist_directory = '/app/chroma_data'
        embedding = HuggingFaceEmbeddings(
            #Cria o gerador de embeddings com o modelo 'all-mpnet-base-v2'
            model_name="sentence-transformers/all-mpnet-base-v2"  # Modelo que gera embeddings de 768 dimensões
        )

        #Cria a base de dados vetorial Chroma com os embeddings
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding,
        )
        #Retorna um mecanismo de pesquisa (retriever) configurado para devolver os 30 documentos mais relevantes
        return vector_store.as_retriever(
            search_kwargs={'k': 30},
        )

    def __build_messages(self, history_messages, question):
        #Constrói a lista de mensagens com base no histórico
        messages = []
        for message in history_messages:
            #Verifica se a mensagem é do utilizador ou do assistente
            message_class = HumanMessage if message.get('fromMe') else AIMessage
            #Adiciona a mensagem com o conteúdo apropriado
            messages.append(message_class(content=message.get('body')))
        #Adiciona a pergunta atual no final da lista de mensagens    
        messages.append(HumanMessage(content=question))
        return messages

    def invoke(self, history_messages, question):
        #Template do sistema com instruções específicas para o assistente
        SYSTEM_TEMPLATE = '''
        Responde as perguntas dos utilizadores com base no contexto abaixo.
        És um assistente especializado em tirar dúvidas sobre recursos humanos de uma empresa.
        Tira dúvidas dos possíveis colaboradores que entram em contato.
        Responde de forma natural, agradável e respeitosa. Sejas objetivo nas respostas, com informações
        claras e diretas. Foca em seres natural e humanizado, como um diálogo comum entre duas pessoas.
        Leva em consideração também o histórico de mensagens da conversa com o utilizador.
        Responde sempre em português europeu e cita as fontes.

        <context>
        {context}
        </context>
        '''

        #Usa o retriever para buscar documentos relevantes à pergunta
        docs = self.__retriever.invoke(question)

        # Cria o template de prompt com mensagens do sistema e da conversa
        question_answering_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    'system',
                    SYSTEM_TEMPLATE,
                ),
                MessagesPlaceholder(variable_name='messages'),
            ]
        )

        #Cria a cadeia (Chains) de documentos que combina contexto + prompt + modelo de chat
        document_chain = create_stuff_documents_chain(self.__chat, question_answering_prompt)
        
        #Invoca o modelo com o contexto e histórico de mensagens
        response = document_chain.invoke(
            {
                'context': docs,
                'messages': self.__build_messages(history_messages, question),
            }
        )

        #Retorna a resposta do assistente
        return response


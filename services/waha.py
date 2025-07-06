#Importa a biblioteca 'requests' para fazer chamadas HTTP
import requests

#Classe Waha: encapsula a comunicação com a API do servidor WAHA (interface com WhatsApp)
class Waha:

    def __init__(self):
        #Define a URL base da API (presume que o serviço 'waha' está disponível via Docker)
        self.__api_url = 'http://waha:3000'

    #Envia uma mensagem de texto para um contacto via WhatsApp
    def send_message(self, chat_id, message):
        url = f'{self.__api_url}/api/sendText' #Endpoint para envio de mensagens
        headers = {
            'Content-Type': 'application/json',
        }
        payload = {
            'session': 'default',   #Nome da sessão (deve corresponder à configurada no WAHA)
            'chatId': chat_id,      #Número do contacto (ex: '351912345678@c.us')
            'text': message,        #Texto da mensagem a enviar
        }
        #Faz o POST para enviar a mensagem
        requests.post(
            url=url,
            json=payload,
            headers=headers,
        )

    # Recupera o histórico de mensagens com um determinado contacto
    def get_history_messages(self, chat_id, limit):
        url = f'{self.__api_url}/api/default/chats/{chat_id}/messages?limit={limit}&downloadMedia=false'
        headers = {
            'Content-Type': 'application/json',
        }
        #Faz GET ao endpoint de mensagens
        response = requests.get(
            url=url,
            headers=headers,
        )
        return response.json() # Devolve a resposta da API já convertida em dicionário Python

    #Envia um sinal à API a indicar que o bot está a "escrever"
    def start_typing(self, chat_id):
        url = f'{self.__api_url}/api/startTyping'
        headers = {
            'Content-Type': 'application/json',
        }
        payload = {
            'session': 'default',
            'chatId': chat_id,
        }
        requests.post(
            url=url,
            json=payload,
            headers=headers,
        )

    # Envia um sinal à API a indicar que o bot "parou" de "escrever"
    def stop_typing(self, chat_id):
        url = f'{self.__api_url}/api/stopTyping'
        headers = {
            'Content-Type': 'application/json',
        }
        payload = {
            'session': 'default',
            'chatId': chat_id,
        }
        requests.post(
            url=url,
            json=payload,
            headers=headers,
        )

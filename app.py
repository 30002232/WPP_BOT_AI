#Importa as bibliotecas Flask necessárias para criar a API
from flask import Flask, request, jsonify

#Importa a classe AIBot, que faz a integração com LLM + RAG
from bot.ai_bot import AIBot

#Importa a classe Waha, que envia e recebe mensagens via WhatsApp (WAHA)
from services.waha import Waha

#Inicializa a aplicação Flask
app = Flask(__name__)

@app.route('/chatbot/webhook/', methods=['POST'])
def webhook():
    try:
        #Extrai o JSON enviado no corpo do POST
        data = request.json

        # Extrai o ID do remetente e a mensagem recebida
        chat_id = data['payload']['from']
        received_message = data['payload']['body']

        # Verifica se a mensagem veio de um grupo (ignora grupos)
        is_group = '@g.us' in chat_id

        if is_group:
            return jsonify({'status': 'success', 'message': 'Mensagem de grupo ignorada.'}), 200

        #Inicializa instâncias do WAHA e do bot de IA
        waha = Waha()
        ai_bot = AIBot()

        waha.start_typing(chat_id=chat_id) #Mostra "a escrever..." no WhatsApp

        #Procura o histórico das últimas 10 mensagens para fornecer contexto
        history_messages = waha.get_history_messages(chat_id=chat_id, limit=10)

        #Envia a nova pergunta + histórico ao modelo de IA
        response_message = ai_bot.invoke(history_messages=history_messages, question=received_message)

        #Extrai o conteúdo textual da resposta gerada
        message_text = response_message.content if hasattr(response_message, "content") else str(response_message)

        #Envia a resposta gerada para o utilizador via WhatsApp
        waha.send_message(chat_id=chat_id, message=message_text)

        #Finaliza o estado de digitação
        waha.stop_typing(chat_id=chat_id)

        #Retorna sucesso ao WAHA
        return jsonify({'status': 'success'}), 200

    except Exception as e:
        #Log do erro no terminal
        print(f"[webhook] Erro: {str(e)}")

        #Retorna um erro 500 com a descrição
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    #Inicia o servidor Flask localmente em modo debug
    app.run(host='0.0.0.0', port=5000, debug=True)

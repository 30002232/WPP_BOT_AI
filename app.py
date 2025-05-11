from flask import Flask, request, jsonify

from bot.ai_bot import AIBot
from services.waha import Waha

app = Flask(__name__)

@app.route('/chatbot/webhook/', methods=['POST'])
def webhook():
    try:
        data = request.json
        chat_id = data['payload']['from']
        received_message = data['payload']['body']
        is_group = '@g.us' in chat_id

        if is_group:
            return jsonify({'status': 'success', 'message': 'Mensagem de grupo ignorada.'}), 200

        waha = Waha()
        ai_bot = AIBot()

        waha.start_typing(chat_id=chat_id)
        history_messages = waha.get_history_messages(chat_id=chat_id, limit=10)
        response_message = ai_bot.invoke(history_messages=history_messages, question=received_message)

        message_text = response_message.content if hasattr(response_message, "content") else str(response_message)

        waha.send_message(chat_id=chat_id, message=message_text)
        waha.stop_typing(chat_id=chat_id)

        return jsonify({'status': 'success'}), 200

    except Exception as e:
        print(f"[webhook] Erro: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

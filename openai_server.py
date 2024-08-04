from flask import Flask, request, jsonify, Response, json
from time import sleep
from smart.graph import chatBot
from smart.helpers.graph_state import GraphState
from smart.helpers.generalHelpers import escape_messages_curly_braces,escape_curly_braces
app = Flask(__name__)


def get_answer(query,messages, update_callback):
    state = GraphState()
    print("query: ", query)
    state["prompt"] = escape_curly_braces(query["content"])
    state["previous_result"] = "None"
    state["previous_code"] = "None"
    state["failedTimes"] = 0
    state["type"] = "other"
    state["messages"] = escape_messages_curly_braces(messages)
    state["links"] = []
    state["additionalResources"] = []
    #state["update_process"] = update_callback
    #update_callback("Starting the process...\n")
    
    result = chatBot.invoke(state)
    
    return result["answer"]
@app.route('/openapi/v1/models')
def openapi_models():
    response_data = {
        "object": "list",
        "data": [
            {
            "id": "smartass-llm-8b-fp16",
            "object": "model",
            "created": 1686935002,
            "owned_by": "Termix corp"
            }
        ]
    }
    return jsonify(response_data)

def generate_template(result):
    return {
        "id": "chatcmpl",
        "model": "gift-llm-8b-fp16",
        "object": "chat.completion.chunk",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": result
                },
            }
        ],
    }


@app.route('/openapi/v1/chat/completions', methods=["POST"])
def openapi_chat():
    data = request.get_json()
    messages = data.get("messages", "default_value")
    messagesFormated = [(message["role"], message["content"]) for message in messages]

    def generate():
        message = messages[-1]
        def update_progress(update):
            splitto4 = len(update) // 4
            for i in range(4):
                yield f"data: {json.dumps(generate_template(update[i*splitto4:(i+1)*splitto4]))}\n\n"
        #yield from update_progress("Working on it...")
        

        result = get_answer(message, messagesFormated, update_progress)
        response_data = generate_template(result)
        yield f"data: {json.dumps(response_data)}\n\n"

    return Response(generate(), content_type="text/event-stream")

@app.route('/chat', methods=['GET'])
def handle_get_request():
    # Process the incoming GET request
    data = request.get_json()
    query = data.get('query', 'default_value')
    messages = data.get("messages", "default_value")

    message = messages[-1]
    result = get_answer(message)
    
    # Create a response

    # Send the response in JSON format
    return jsonify(generate_template(result))

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)

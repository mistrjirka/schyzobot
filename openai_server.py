from flask import Flask, request, jsonify, Response, json
from time import sleep
from smart.graph import chatBot
from smart.helpers.graph_state import GraphState, STATUS_BLOCK_START, STATUS_BLOCK_END
from smart.helpers.generalHelpers import escape_messages_curly_braces,escape_curly_braces
from queue import Queue
from threading import Thread
from flask import stream_with_context

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
    messages = data.get("messages", [])
    stream = data.get("stream", False)
    print(f"Stream mode: {stream}")
    messages_formatted = [(message["role"], message["content"]) for message in messages]
    message = messages[-1]

    if stream:
        def generate():
            for chunk in get_answer_stream(message, messages_formatted):
                json_data = json.dumps(chunk)
                yield f'data: {json_data}\n\n'
            yield 'data: [DONE]\n\n'

        return Response(stream_with_context(generate()), mimetype='text/event-stream')
    else:
        result = get_answer(message, messages_formatted)
        response_data = generate_template(result)
        return jsonify(response_data)

def filter_status_messages(messages):
    """Filter out status update messages that start with emojis"""
    return [msg for msg in messages if not any(emoji in msg[1] for emoji in ["🔍", "🌐", "📚", "✅", "💭", "🤔", "✨", "🚀"])]

def get_answer_stream(query, messages):
    print("Starting stream processing...")  # Debug print at the very start
    state = GraphState()
    state["prompt"] = escape_curly_braces(query["content"])
    state["failedTimes"] = 0
    state["type"] = "other"
    # Filter messages before setting them in state
    filtered_messages = filter_status_messages(messages)
    state["messages"] = escape_messages_curly_braces(filtered_messages)
    state["links"] = []
    state["additionalResources"] = []

    progress_queue = Queue()

    def update_callback(message):
        print(f"Progress update: {message}")
        progress_queue.put(message)

    # Start status block at the beginning
    update_callback(STATUS_BLOCK_START)
    state["update_process"] = update_callback

    def run_chatbot():
        print("Chatbot thread started")  # Debug print
        result = chatBot.invoke(state)
        print("Chatbot processing completed")  # Debug print
        progress_queue.put(None)

    thread = Thread(target=run_chatbot)
    thread.start()

    role_sent = False
    chunk_counter = 0  # Add counter for debugging
    while True:
        progress_message = progress_queue.get()
        if progress_message is None:
            print("Received completion signal")  # Debug print
            break

        delta = {}
        if not role_sent:
            delta["role"] = "assistant"
            role_sent = True
        delta["content"] = progress_message

        chunk = {
            "id": f"chatcmpl-{chunk_counter}",  # Add unique ID for each chunk
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "delta": delta,
                    "index": 0,
                    "finish_reason": None
                }
            ]
        }
        print(f"Sending chunk {chunk_counter}: {json.dumps(chunk)}")  # Debug print
        chunk_counter += 1
        yield chunk

    # Send the final answer in chunks
    final_answer = state.get("answer", "")
    print(f"Sending final answer of length: {len(final_answer)}")  # Debug print
    
    for chunk_text in split_text(final_answer):
        chunk = {
            "id": f"chatcmpl-{chunk_counter}",
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "delta": {"content": chunk_text},
                    "index": 0,
                    "finish_reason": None
                }
            ]
        }
        print(f"Sending final chunk {chunk_counter}: {json.dumps(chunk)}")  # Debug print
        chunk_counter += 1
        yield chunk

    print("Sending completion signal")  # Debug print
    yield {
        "id": f"chatcmpl-{chunk_counter}",
        "object": "chat.completion.chunk",
        "choices": [
            {
                "delta": {},
                "index": 0,
                "finish_reason": "stop"
            }
        ]
    }

def split_text(text, max_length=50):
    # Add debug print for text splitting
    chunks = [text[i:i + max_length] for i in range(0, len(text), max_length)]
    print(f"Split text into {len(chunks)} chunks")  # Debug print
    return chunks

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

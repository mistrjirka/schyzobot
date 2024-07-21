from flask import Flask, request, jsonify, Response
import json
import time

app = Flask(__name__)

@app.route('/v1/models', methods=['GET'])
def list_models():
    models = {
        "data": [
            {"id": "text-davinci-003", "object": "model", "owned_by": "openai"},
            {"id": "gpt-3.5-turbo", "object": "model", "owned_by": "openai"}
        ],
        "object": "list"
    }
    return jsonify(models)

@app.route('/v1/completions', methods=['POST'])
def create_completion():
    data = request.json
    response = {
        "id": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
        "object": "text_completion",
        "created": int(time.time()),
        "model": data.get("model"),
        "choices": [
            {
                "text": "This is a dummy response for your prompt.",
                "index": 0,
                "logprobs": None,
                "finish_reason": "length"
            }
        ],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 7,
            "total_tokens": 12
        }
    }
    return jsonify(response)

@app.route('/v1/chat/completions', methods=['POST'])
def create_chat_completion():
    data = request.json
    if data.get("stream", False):
        def generate():
            for i in range(5):
                chunk = {
                    "id": f"chatcmpl-{i}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": data.get("model"),
                    "choices": [
                        {"delta": {"content": f" This is part {i+1} of the response."}, "index": 0}
                    ]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                time.sleep(1)
        return Response(generate(), content_type='text/event-stream')
    else:
        response = {
            "id": "chatcmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": data.get("model"),
            "choices": [
                {"message": {"role": "assistant", "content": "This is a dummy chat completion response."}, "index": 0}
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}
        }
        return jsonify(response)

if __name__ == '__main__':
    app.run(port=5000)

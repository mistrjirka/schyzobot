import unittest
import openai
import threading
import time

# Assuming the dummy server code is in a file named dummy_openai_server.py
from openai_server import app

class TestDummyOpenAIServerWithOpenAI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Run the Flask app in a separate thread
        cls.server = threading.Thread(target=app.run, kwargs={'port': 5000})
        cls.server.setDaemon(True)
        cls.server.start()
        time.sleep(1)  # Give the server time to start

        # Configure OpenAI to use the local server
        openai.api_base = 'http://localhost:5000/v1'
        openai.api_key = 'sk_test_123'

    def test_list_models(self):
        response = openai.Model.list()
        self.assertGreater(len(response['data']), 0)

    def test_create_completion(self):
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt="Tell me a joke",
            max_tokens=5
        )
        self.assertEqual(response['choices'][0]['text'], "This is a dummy response for your prompt.")

    def test_create_chat_completion(self):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Tell me a story"}],
            stream=False
        )
        self.assertEqual(response['choices'][0]['message']['content'], "This is a dummy chat completion response.")

    def test_create_chat_completion_stream(self):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Tell me a story"}],
            stream=True
        )
        chunks = []
        for chunk in response:
            chunks.append(chunk)
        self.assertGreater(len(chunks), 0)
        for i, chunk in enumerate(chunks):
            self.assertIn('choices', chunk)
            self.assertIn('delta', chunk['choices'][0])
            self.assertEqual(chunk['choices'][0]['delta']['content'], f" This is part {i+1} of the response.")

if __name__ == '__main__':
    unittest.main()

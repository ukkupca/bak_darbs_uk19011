import sys
import json
import openai
import pinecone
import tiktoken
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

openai.api_key = "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"
pinecone.init(api_key="a433ce32-e7cb-4684-9280-1d201daccc85", environment="eu-west1-gcp")
index = pinecone.Index("virtual-agent-v0")


def count_tokens(text):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))


def gpt_embedding(content, engine='text-embedding-ada-002'):
    content = content.encode(encoding='ASCII', errors='ignore').decode()
    res = openai.Embedding.create(input=content, engine=engine)
    vector = res['data'][0]['embedding']
    return vector


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)


def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return json.load(infile)


def save_json(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        json.dump(content, outfile, ensure_ascii=False, sort_keys=True, indent=2)


def load_history(results, user_type):
    matches = [result for result in results['matches']]
    result_data = list()
    for m in matches:
        if m['metadata']['user'] == user_type:
            data = list()
            date = m['metadata']['time'].strftime("%m/%d/%Y, %H:%M:%S")
            data.append("Time: " + date)
            data.append("User: " + m['metadata']['user'])
            data.append("Message: " + m['metadata']['message'])
            data = '\n'.join(data)
            result_data.append(data)
    message_block = '\n'.join(result_data).strip()
    return message_block


conversation_history = []

# The main loop for the chatbot
while True:
    payload = list()

    # Getting user input, time and vectorizing them for the index
    chat_input = input("You: ")
    conversation_history.append({"role": "user", "content": chat_input})
    date_time = datetime.now()
    user_date_time = date_time.now().strftime("%m/%d/%Y, %H:%M:%S")
    metadata = {
        'time': user_date_time,
        'user': 'USER',
        'message': chat_input,
    }
    user_message = '%s: %s - %s' % ('USER', user_date_time, chat_input)
    user_message_vector = gpt_embedding(user_message)
    identity = str(uuid4())
    payload.append((identity, user_message_vector, metadata))

    # Getting relevant previous messages from index
    # top_k sets how many results will be returned
    index_history = index.query(vector=user_message_vector, top_k=100, include_values=False, include_metadata=True)
    user_history = load_history(index_history, 'user')
    agent_history = load_history(index_history, 'EVE')
    prompt = open_file('prompt_config').replace('<<USER>>', user_history).replace('<<EVE>>', agent_history)

    # Preparing prompt structure for API
    messages = conversation_history.copy()
    messages.insert(0, {"role": "system", "content": prompt})

    # Making API call to OpenAI with the prompt
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=True,
    )

    full_response = ""
    sys.stdout.write("Response: ")
    # Process the response from the API and stream it
    for chunk in response:
        if "role" in chunk["choices"][0]["delta"]:
            continue

        elif "content" in chunk["choices"][0]["delta"]:
            r_text = chunk["choices"][0]["delta"]["content"]
            full_response += r_text
            sys.stdout.write(r_text)
            sys.stdout.flush()
    print()

    # Saving response to index
    date_time = datetime.now()
    agent_date_time = date_time.strftime("%m/%d/%Y, %H:%M:%S")
    agent_metadata = {
        'time': agent_date_time,
        'user': 'EVE',
        'message': full_response,
    }
    agent_message = '%s: %s - %s' % ('EVE', agent_date_time, full_response)
    agent_message_vector = gpt_embedding(agent_message)
    identity = str(uuid4())
    payload.append((identity, agent_message_vector, agent_metadata))

    # Adding response to current conversation history
    conversation_history.append({"role": "assistant", "content": full_response})
    messages.append({"role": "assistant", "content": full_response})
    save_json('logs/%s.json' % time.time(), messages)

    # Uploading new data to index
    index.upsert(payload)

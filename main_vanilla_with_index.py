import sys
import openai
import pinecone
import time
import os
import common
import datetime
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv('KEY_OPEN_AI')
pinecone.init(api_key=os.getenv('KEY_PINECONE'), environment=os.getenv('ENV_PINECONE'))
index = pinecone.Index("virtual-agent-v0")


def load_history_old(results):
    matches = [result for result in results['matches']]
    result_data = list()
    for m in matches:
        data = list()
        timestamp = m['metadata']['timestamp']
        date_time = datetime.datetime.fromtimestamp(int(timestamp))
        data.append("Time: " + date_time.strftime('%Y-%m-%d %H:%M:%S'))
        data.append("Text: " + m['metadata']['content'])
        data = '\n'.join(data)
        result_data.append(data)
    data_without_duplicates = list(set(result_data))
    message_block = '\n'.join(data_without_duplicates).strip()
    return message_block


# Main

current_conversation_history = []
while True:
    user_payload = list()
    agent_payload = list()
    # Getting user input, adding to local history
    chat_input = input("You: ")
    current_conversation_history.append({"role": "user", "content": chat_input})

    # Format for index
    identity = str(int(time.time()))
    user_message_vector = common.gpt_embedding(chat_input)
    metadata = {
        'timestamp': identity,
        'content': chat_input,
    }
    user_payload.append((identity, user_message_vector, metadata))

    # Getting relevant previous messages from index
    # top_k sets how many results will be returned
    user_history = index.query(namespace='USER',
                               vector=user_message_vector,
                               top_k=40,
                               include_values=False,
                               include_metadata=True)
    agent_history = index.query(namespace='AGENT',
                                vector=user_message_vector,
                                top_k=10,
                                include_values=False,
                                include_metadata=True)

    loaded_user_history = load_history_old(user_history)
    loaded_agent_history = load_history_old(user_history)

    prompt = common.open_file('prompt-configs/default_system_config') \
        .replace('<<USER>>', loaded_user_history) \
        .replace('<<AGENT>>', loaded_agent_history) \
        .replace('<<CURRENT>>', chat_input)

    # Preparing prompt structure for API
    messages = current_conversation_history.copy()
    messages.insert(0, {"role": "system", "content": prompt})

    # Making API call to OpenAI with the prompt
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=True,
        temperature=0
    )

    # Processing the response
    sys.stdout.write("Eve: ")
    full_response = common.get_response(response, True)

    # Adding response to local current conversation history
    current_conversation_history.append({"role": "assistant", "content": full_response})

    # Saving a local log of what API has received and what was the answer
    messages.append({"role": "assistant", "content": full_response})
    common.save_json('logs/%s.json' % int(time.time()), messages)

    # Format for index
    identity = str(int(time.time()))
    agent_message_vector = common.gpt_embedding(full_response)
    agent_metadata = {
        'timestamp': identity,
        'content': full_response,
    }
    agent_payload.append((identity, agent_message_vector, agent_metadata))

    # Uploading new data to index
    index.upsert(user_payload, "USER")
    index.upsert(agent_payload, "AGENT")

import sys
import openai
import pinecone
import time
import os
import common
import datetime
from dotenv import load_dotenv
from uuid import uuid4

import process_index

load_dotenv()
openai.api_key = os.getenv('KEY_OPEN_AI')
pinecone.init(api_key=os.getenv('KEY_PINECONE'), environment=os.getenv('ENV_PINECONE'))
index = pinecone.Index("virtual-agent-v0")


def load_history_old(results, user_type):
    matches = [result for result in results['matches']]
    result_data = list()
    for m in matches:
        if m['metadata']['user'] == user_type:
            data = list()
            timestamp = m['metadata']['timestamp']
            date_time = datetime.datetime.fromtimestamp(timestamp)
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
    payload = list()

    # Getting user input, adding to local history
    chat_input = input("You: ")
    current_conversation_history.append({"role": "user", "content": chat_input})

    # Format for index
    identity = str(uuid4())
    user_message_vector = common.gpt_embedding(chat_input)
    metadata = {
        'timestamp': int(time.time()),
        'message': chat_input,
    }
    payload.append((identity, user_message_vector, metadata))

    # Base conversation prompt
    base_prompt = common.open_file('prompt-configs/default_system_config')

    # Preparing prompt structure for API
    messages = current_conversation_history.copy()
    messages.insert(0, {"role": "system", "content": base_prompt})

    # Making API call to OpenAI with the prompt
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=True,
        temperature=0
    )

    # give messages to agent, so it can decide whether the response is ok, or DB should be called

    # Getting relevant previous messages from index
    # top_k sets how many results will be returned
    index_user_history = index.query(namespace='USER',
                                     vector=user_message_vector,
                                     top_k=100,
                                     include_values=False,
                                     include_metadata=True)

    agent_user_history = index.query(namespace='AGENT',
                                     vector=user_message_vector,
                                     top_k=100,
                                     include_values=False,
                                     include_metadata=True)

    user_history = process_index.load_user_type_history(index_user_history, 'USER')
    agent_history = process_index.load_user_type_history(agent_user_history, 'AGENT')

    user_history_task = ""
    processed_user_history = process_index.process_history(user_history, user_history_task, 'USER')
    agent_history_task = ""
    processed_agent_history = process_index.process_history(agent_history, agent_history_task, 'AGENT')


    prompt = common.open_file('prompt-configs/default_system_config')\
        .replace('<<USER>>', user_history)\
        .replace('<<EVE>>', agent_history)\
        .replace('<<CURRENT>>', chat_input)



    # Making API call to OpenAI with the prompt
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=True,
        temperature=0
    )

    # Processing the response
    sys.stdout.write("Response: ")
    full_response = common.get_response(response, True)

    # Adding response to local current conversation history
    current_conversation_history.append({"role": "assistant", "content": full_response})

    # Saving a local log of what API has received and what was the answer
    messages.append({"role": "assistant", "content": full_response})
    common.save_json('logs/%s.json' % int(time.time()), messages)

    # Format for index
    identity = str(uuid4())
    agent_message_vector = common.gpt_embedding(full_response)
    agent_metadata = {
        'user': 'EVE',
        'content': full_response,
    }
    payload.append((identity, agent_message_vector, agent_metadata))

    # Uploading new data to index
    index.upsert(payload)

import sys
import openai
import pinecone
import time
import os
import common
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
            data.append("Speaker: " + m['metadata']['user'])
            data.append("Message: " + m['metadata']['message'])
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
        'user': 'USER',
        'message': chat_input,
    }
    payload.append((identity, user_message_vector, metadata))

    # Getting relevant previous messages from index
    # top_k sets how many results will be returned
    index_history = index.query(vector=user_message_vector, top_k=100, include_values=False, include_metadata=True)
    user_history = process_index.load_history(index_history, 'USER')
    agent_history = process_index.load_history(index_history, 'EVE')

    # s = process_index.history_to_string(user_history)

    processed_user_history = process_index.process_history(user_history)
    # History can be too long to pass to LLM
    # Implement a mechanism that cycles through messages
    # Pass instruction prompt + users question + batch of messages
    # Process all batches
    # Get answer?

    prompt = common.open_file('prompt-configs/vanilla_prompt_config')\
        .replace('<<USER>>', user_history)\
        .replace('<<EVE>>', agent_history)\
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
        'message': full_response,
    }
    payload.append((identity, agent_message_vector, agent_metadata))

    # Uploading new data to index
    index.upsert(payload)

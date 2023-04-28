import time
import common


def prepare_and_add(chat_input, payload):
    identity = str(int(time.time()))
    message_vector = common.gpt_embedding(chat_input)
    metadata = {
        'timestamp': identity,
        'content': chat_input,
    }
    payload.append((identity, message_vector, metadata))


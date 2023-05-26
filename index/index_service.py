import time
import common


def prepare_and_add(chat_input, payload, timestamp=None):
    if timestamp is None:
        identity = str(int(time.time()))
    else:
        identity = timestamp
    message_vector = common.gpt_embedding(chat_input)
    metadata = {
        'timestamp': identity,
        'content': chat_input,
    }
    payload.append((identity, message_vector, metadata))


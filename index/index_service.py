import time
import common


def prepare_payload(chat_input, payload):
    # Format for index
    identity = str(int(time.time()))
    message_vector = common.gpt_embedding(chat_input)
    # TODO: Possibly could upload only unique messages
    metadata = {
        'timestamp': identity,
        'content': chat_input,
    }
    payload.append((identity, message_vector, metadata))

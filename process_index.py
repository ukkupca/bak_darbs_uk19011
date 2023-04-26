import common

MAX_MEMORY_TOKENS = 4097
def load_history(results, user_type):
    history = []
    matches = [r for r in results['matches'] if r['metadata']['user'] == user_type]
    for m in matches:
        history.append({"role": user_type, "content": m['metadata']['message']})
    return history


def history_to_string(history):
    return '\n'.join(history).strip()


def split_dictionary(input_dict, chunk_size):
    res = []
    new_dict = {}
    for k, v in input_dict.items():
        if len(new_dict) < chunk_size:
            new_dict[k] = v
        else:
            res.append(new_dict)
            new_dict = {k: v}
    res.append(new_dict)
    return res


def process_history(history):
    # Calculate the total tokens in the conversation history
    total_tokens = sum(common.count_tokens(message["content"]) for message in history)
    entry_count = len(history)
    history_batches = []
    if total_tokens > MAX_MEMORY_TOKENS:
        batches = split_dictionary(history, entry_count/2)


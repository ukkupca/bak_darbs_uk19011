import common
import datetime
import openai

MAX_MEMORY_TOKENS = 4097
BATCH_MEMORY_TOKENS = 4000


def load_user_type_history(results, user_type):
    history = []
    for m in results['matches']:
        timestamp = m['metadata']['timestamp']
        date_time = datetime.datetime.fromtimestamp(timestamp)
        content = 'Time: %s Content: %s' % (date_time.strftime('%Y-%m-%d %H:%M:%S'), m['metadata']['message'])
        history.append(
            {
                "role": user_type,
                "content": content,
            }
        )
    return history


def history_to_string(history):
    return '\n'.join(history).strip()  # not correct


def remove_duplicates(array):
    unique_data = []
    for item in array:
        content = strip_until_content(item['content'])
        if content not in unique_data:
            unique_data.append(item)
    return unique_data


def strip_until_content(text):
    parts = text.split("Content:", 1)
    if len(parts) > 1:
        return parts[1].strip()
    return text


def split_array(array, limit=BATCH_MEMORY_TOKENS, current_batch=None, result=None):
    if current_batch is None:
        current_batch = []
    if result is None:
        result = []

    if not array:
        if current_batch:
            result.append(current_batch)
        return result

    item = array.pop(0)
    item_tokens = common.count_tokens(item["content"])

    if item_tokens >= limit:
        raise ValueError("An item in the array has tokens count exceeding the limit.")

    current_batch_tokens = sum(common.count_tokens(item["content"]) for item in current_batch)

    if current_batch_tokens + item_tokens < limit:
        current_batch.append(item)
    else:
        result.append(current_batch)
        current_batch = [item]

    return split_array(array, limit, current_batch, result)


def get_contents_as_string(batch):
    contents = []
    for b in batch:
        contents.append(b['content'])
    return '\n'.join(contents)


def process_history(history, task, user):
    result = []
    duplicates_removed = remove_duplicates(history)
    batches = split_array(duplicates_removed)
    # TODO: write system prompt
    system_prompt = common.open_file('prompt-configs/batching_system_config')
    system_message = {
        "role": "system",
        "content": system_prompt
    }
    prompt = common.open_file('prompt-configs/batching_prompt_config')\
        .replace('<<USER_TYPE>>', user)\
        .replace('<<TASK>>', task)
    for b in batches:
        batch_prompt = prompt.replace('<<HISTORY>>', b.get_contents_as_string())
        user_message = {
            "role": "user",
            "content": batch_prompt
        }
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[system_message, user_message],
            stream=True,
            temperature=0
        )
        result.append(common.get_response(response))
    return result




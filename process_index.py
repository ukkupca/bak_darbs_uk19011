import common
import datetime
import openai

import env_loader

MAX_MEMORY_TOKENS = 4097
BATCH_MEMORY_TOKENS = 4000


def load_user_type_history(results, user_type):
    history = []
    for m in results['matches']:
        timestamp = m['metadata']['timestamp']
        date_time = datetime.datetime.fromtimestamp(int(timestamp))
        content = 'Time: %s Content: %s' % (date_time.strftime('%Y-%m-%d %H:%M:%S'), m['metadata']['message'])
        history.append(
            {
                "role": user_type,
                "content": content,
            }
        )
    return history


def remove_duplicates(array):
    unique_data = []
    result = []
    for item in array:
        content = strip_until_content(item['content'])
        if content not in unique_data:
            unique_data.append(item['content'])
            result.append(item)
    return result


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
        timestamp = b['timestamp']
        date_time = datetime.datetime.fromtimestamp(int(timestamp))
        contents.append('On %s: %s' % (date_time, b['content']))
    return '\n'.join(contents)


def process_user_messages(history, query):
    system_prompt = common.open_file('batching-prompt-configs/batching_system_config__user').replace('<<QUERY>>', query)
    user_prompt = common.open_file('batching-prompt-configs/batching_prompt_config')
    return process_history(history, system_prompt, user_prompt)


def process_agent_messages(history, query):
    system_prompt = common.open_file('batching-prompt-configs/batching_system_config__agent').replace('<<QUERY>>', query)
    user_prompt = common.open_file('batching-prompt-configs/batching_prompt_config')
    return process_history(history, system_prompt, user_prompt)


def process_summaries(summaries, query):
    system_prompt = common.open_file('batching-prompt-configs/summary_batching_system_config').replace('<<QUERY>>', query)
    user_prompt = common.open_file('batching-prompt-configs/summary_batching_prompt_config')
    return process_history(summaries, system_prompt, user_prompt)


def process_history(history, system_prompt, user_prompt):
    if not history['matches']:
        return ""

    result = []
    metadata = [m['metadata'] for m in history['matches']]
    duplicates_removed = remove_duplicates(metadata)
    batches = split_array(duplicates_removed)
    system_message = {
        "role": "system",
        "content": system_prompt
    }

    for b in batches:
        batch_prompt = user_prompt.replace('<<HISTORY>>', get_contents_as_string(b))
        user_message = {
            "role": 'user',
            "content": batch_prompt
        }
        response = openai.ChatCompletion.create(
            model=env_loader.openai_model,
            messages=[system_message, user_message],
            stream=True,
            temperature=0
        )
        result.append(common.get_response(response))
    return '\n'.join(result)

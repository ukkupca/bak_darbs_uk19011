import common
import datetime
import openai

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


def history_to_string(history):
    return '\n'.join(history).strip()  # not correct


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


def process_history(history, query, user):
    if not history['matches']:
        return ""

    result = []
    metadata = [m['metadata'] for m in history['matches']]
    duplicates_removed = remove_duplicates(metadata)
    batches = split_array(duplicates_removed)
    system_prompt = get_system_prompt(user)\
        .replace('<<USER_TYPE>>', user)\
        .replace('<<QUERY>>', query)
    system_message = {
        "role": "system",
        "content": system_prompt
    }

    prompt = common.open_file('prompt-configs/batching_prompt_config')
    role = get_role(user)

    for b in batches:
        batch_prompt = prompt.replace('<<HISTORY>>', get_contents_as_string(b))
        user_message = {
            "role": role,
            "content": batch_prompt
        }
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[system_message, user_message],
            stream=True,
            temperature=0
        )
        result.append(common.get_response(response))
    return '\n'.join(result)


def get_system_prompt(user):
    if user == 'USER':
        prompt = common.open_file('prompt-configs/batching_prompt_config')
    else:
        prompt = common.open_file('prompt-configs/agent_batching_system_config')
    return prompt


def get_role(user):
    if user == 'USER':
        role = 'user'
    else:
        role = 'assistant'
    return role




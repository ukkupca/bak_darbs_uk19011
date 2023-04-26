import tiktoken
import json
import openai
import sys


def count_tokens(text):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))


def gpt_embedding(content, engine='text-embedding-ada-002'):
    content = content.encode(encoding='ASCII', errors='ignore').decode()
    res = openai.Embedding.create(input=content, engine=engine)
    vector = res['data'][0]['embedding']
    return vector


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)


def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return json.load(infile)


def save_json(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        json.dump(content, outfile, ensure_ascii=False, sort_keys=True, indent=2)


def get_response(response, stream=False):
    full_response = ""
    # Streaming response to terminal
    for chunk in response:
        if "role" in chunk["choices"][0]["delta"]:
            continue

        elif "content" in chunk["choices"][0]["delta"]:
            r_text = chunk["choices"][0]["delta"]["content"]
            full_response += r_text
            if stream:
                sys.stdout.write(r_text)
                sys.stdout.flush()
    print()
    return full_response

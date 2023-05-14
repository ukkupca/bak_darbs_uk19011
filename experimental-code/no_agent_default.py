import openai
import tiktoken

openai.api_key = "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"


def count_tokens(text):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))


# Set the maximum token limit
MAX_MEMORY_TOKENS = 50

# Initialize the conversation_history list
conversation_history = [
    {"role": "system", "content": "You are a helpful assistant."}
]

# The main loop for the chatbot
while True:
    # Get user input
    chat_input = input("You: ")

    # # Check if the user wants to print the entire chat history
    # if chat_input.lower() == "print chat history":
    #     print_chat_history = True
    # else:
    #     print_chat_history = False

    # Append user input to the conversation history
    conversation_history.append({"role": "user", "content": chat_input})

    # Calculate the total tokens in the conversation history
    total_tokens = sum(count_tokens(message["content"]) for message in conversation_history)
    total_tokens_before_removal = total_tokens

    # Remove the oldest message from conversation history if total tokens exceed the maximum limit
    while total_tokens > MAX_MEMORY_TOKENS:
        if len(conversation_history) > 2:
            # 0 element is system message, so we don't remove it
            removed_message = conversation_history.pop(1)
            total_tokens -= count_tokens(removed_message["content"])
            # print total tokens used after removing the oldest message

        else:
            break
        print(f"Total tokens before removal: {total_tokens_before_removal}")
        print(f"Total tokens after removal: {total_tokens}")

    # Make API calls to OpenAI with the conversation history and use streaming responses
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation_history,
        stream=True,
    )

    full_response = "Response: "  # for streaming style: sys.stdout.write("Response: ")
    # Process the response from the API
    for chunk in response:
        if "role" in chunk["choices"][0]["delta"]:
            continue

        elif "content" in chunk["choices"][0]["delta"]:
            r_text = chunk["choices"][0]["delta"]["content"]
            conversation_history.append({"role": "assistant", "content": r_text})
            full_response += r_text  # for streaming style: sys.stdout.write(r_text)
            # enable for streaming style: sys.stdout.flush()
    print(full_response)

    # new line after the assistant's response
    print()

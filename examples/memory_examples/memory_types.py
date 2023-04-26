from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory, ConversationSummaryMemory, ConversationBufferWindowMemory, ConversationSummaryBufferMemory, ConversationEntityMemory, CombinedMemory
import os
from pprint import pprint



# NEXT IMPORTS ARE NECESSARY FOR ENTITY MEMORY
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from pydantic import BaseModel
from typing import List, Dict, Any

# NEXT IMPORTS ARE NECESSARY FOR COMBINED MEMORY
from langchain.prompts import PromptTemplate


# defining the llm as OpenAI gpt3
llm = OpenAI(temperature=0.7)

#get the key from the environment variable
os.environ["OPENAI_API_KEY"] = os.getenv("openai_key")

# :::REGULAR MEMORY:::

conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory(k=3)
)

# :::SUMMARY MEMORY:::

# conversation = ConversationChain(
#     llm=llm,
#     verbose=True,
#     memory=ConversationSummaryMemory(llm=llm)
# )

# :::BUFFER WINDOW MEMORY:::

# conversation = ConversationChain(
#     llm=llm,
#     verbose=True,
#     memory=ConversationBufferWindowMemory(k=3)
# )

# :::SUMMARY BUFFER MEMORY::: requires pip install tiktoken
# conversation = ConversationChain(
#     llm=llm,
#     verbose=True,
#     memory=ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
# )

# :::ENTITY MEMORY:::

# conversation = ConversationChain(
#     llm=llm,
#     verbose=True,
#     prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
#     memory=ConversationEntityMemory(llm=llm)
# )

# :::COMBINED MEMORY:::

# conv_memory = ConversationBufferMemory(
#     memory_key="chat_history_lines",
#     input_key="input"
# )

# summary_memory = ConversationSummaryMemory(llm=OpenAI(), input_key="input")
# # Combined
# memory = CombinedMemory(memories=[conv_memory, summary_memory])
# _DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

# Summary of conversation:
# {history}
# Current conversation:
# {chat_history_lines}
# Human: {input}
# AI:"""
# PROMPT = PromptTemplate(
#     input_variables=["history", "input", "chat_history_lines"], template=_DEFAULT_TEMPLATE
# )
# llm = OpenAI(temperature=0)
# conversation = ConversationChain(
#     llm=llm,
#     verbose=True,
#     memory=memory,
#     prompt=PROMPT
# )


# :::MAIN LOOP:::
while True:
    user_input = input("You: ")
    if user_input == "mem" and conversation.verbose == False:
        conversation.verbose = True
        continue
    elif user_input == "mem" and conversation.verbose == True:
        conversation.verbose = False
        continue
    ai = conversation.predict(input=user_input) # or use run without "input:"" ai = conversation.run(user_input)
    print("AI: ", ai)
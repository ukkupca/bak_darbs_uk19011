from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory, ConversationSummaryMemory, \
    ConversationBufferWindowMemory, ConversationSummaryBufferMemory, ConversationEntityMemory, CombinedMemory
import os

# entity memory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from pydantic import BaseModel
from typing import List, Dict, Any

# combined memory
from langchain.prompts import PromptTemplate


llm = ChatOpenAI(
    temperature=0,
    openai_api_key="",
    model_name="gpt-3.5-turbo"
)

# regular memory - all messages are in the buffer
conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory()
)

while True:
    user_input = input("You: ")
    if user_input == "mem" and conversation.verbose == False:
        conversation.verbose = True
        continue
    elif user_input == "mem" and conversation.verbose == True:
        conversation.verbose = False
        continue
    ai = conversation.predict(input=user_input)  # or use run without "input:"" ai = conversation.run(user_input)
    print("AI: ", ai)

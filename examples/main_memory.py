import openai
import pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory

pinecone.init(api_key="a433ce32-e7cb-4684-9280-1d201daccc85", environment="eu-west1-gcp")
# index = pinecone.Index("virtual-agent-v0") NOTNOW_TODO: new index
llm = ChatOpenAI(
    temperature=0,
    openai_api_key="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",
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


def gpt_embedding(content, engine='text-embedding-ada-002'):
    content = content.encode(encoding='ASCII', errors='ignore').decode()
    response = openai.Embedding.create(input=content, engine=engine)
    vector = response['data'][0]['embedding']
    return vector

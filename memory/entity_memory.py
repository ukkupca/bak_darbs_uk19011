import time

from langchain.llms import OpenAI
from langchain.memory import ConversationEntityMemory

import common
import env_loader as e


class EntityMemory:
    llm = OpenAI(
        temperature=0,
        openai_api_key=e.openai_api_key
    )
    memory = ConversationEntityMemory(llm=llm)
    last_user_input = ""
    payload = list()

    def set_last_user_input(self, chat_input):
        self.last_user_input = chat_input
        self.memory.load_memory_variables({"input": self.last_user_input})

    def set_last_agent_input_and_save(self, agent_input):
        self.memory.save_context({"input": self.last_user_input}, {"output": agent_input})

    def prepare_and_add(self, entities):
        for key, entity in entities.items():
            identity = str(int(time.time()))
            message_vector = common.gpt_embedding(key)
            metadata = {
                'timestamp': identity,
                'content': entity,
            }
            self.payload.append((identity, message_vector, metadata))

    def upsert_to_db(self):
        entities = self.memory.entity_store
        self.prepare_and_add(entities.__dict__)
        e.index.upsert(self.payload, 'ENTITY')




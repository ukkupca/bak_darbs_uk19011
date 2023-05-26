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
    memory = ConversationEntityMemory(llm=llm, human_prefix='User')
    last_user_input = ""
    payload = list()

    def set_last_user_input(self, chat_input):
        self.last_user_input = chat_input
        self.memory.load_memory_variables({"input": self.last_user_input})

    def set_last_agent_input_and_save(self, agent_input):
        self.memory.save_context({"input": self.last_user_input}, {"output": agent_input})

    def prepare_and_add(self, entities):
        counter = 0
        for key, entity in entities.items():
            counter = counter + 100
            identity = str(int(time.time()) + counter)
            message_vector = common.gpt_embedding(entity)
            metadata = {
                'timestamp': identity,
                'content': entity,
            }
            self.payload.append((identity, message_vector, metadata))

    def upsert_to_db(self):
        entities = self.memory.entity_store.get_all()
        self.prepare_and_add(entities)
        e.index.upsert(self.payload, 'ENTITY-2')




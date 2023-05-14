from index import index_service
import env_loader as e


class SingleMessageMemory:
    user_payload = list()
    agent_payload = list()

    def add_user_message_to_payload(self, message):
        index_service.prepare_and_add(message, self.user_payload)

    def add_agent_message_to_payload(self, message):
        index_service.prepare_and_add(message, self.agent_payload)

    def upsert_user_payload(self):
        e.index.upsert(self.user_payload, 'USER')

    def upsert_agent_payload(self):
        e.index.upsert(self.agent_payload, 'AGENT')

    def upsert_all_messages(self):
        self.upsert_user_payload()
        self.upsert_agent_payload()

import datetime
import time

from index import index_service
import env_loader as e
import more_itertools as mit


class MessageBatchMemory:
    message_payload = list()
    conversation_log = list()

    def add_user_message(self, message):
        self.add_message_to_memory(message, 'user')

    def add_agent_message(self, message):
        self.add_message_to_memory(message, 'agent')

    def add_message_to_memory(self, message, role):
        data = {
            'timestamp': str(int(time.time())),
            'role': role,
            'content': message
        }
        self.conversation_log.append(data)

    def split_into_batches(self):
        message_count = len(self.conversation_log)
        batch_size = 5
        if message_count <= batch_size:
            return [self.conversation_log]
        batches = mit.windowed(self.conversation_log, batch_size, step=2)
        return list(batches)

    def prepare_payload(self):
        batches = self.split_into_batches()
        counter = 0
        for batch in batches:
            counter = counter + 100
            batch_id = int(batch[0]['timestamp']) + counter
            batch_messages = ""
            for message in batch:
                if message is None:
                    continue
                role = message['role']
                content = message['content']
                batch_messages += f"{role}: {content}\n"
            index_service.prepare_and_add(batch_messages, self.message_payload, str(batch_id))

    def upsert_payload(self):
        e.index.upsert(self.message_payload, 'BATCH')

    def upsert_all_batches(self):
        self.prepare_payload()
        self.upsert_payload()

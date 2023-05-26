import sys

import chats
from logs import log_service
from memory.entity_memory import EntityMemory
from memory.message_batch_memory import MessageBatchMemory
from memory.summary_memory import SummaryMemory

# conversation_logs = log_service.load()
# conversations_summaries = []
# conversations_entities = []

conversation = chats.conversations[6]

# summary = SummaryMemory()
entity = EntityMemory()
# batch = MessageBatchMemory()

counter = 1
for message in conversation:
    counter = counter + 1
    sys.stdout.write("\nAdding message %s" % counter)
    if counter % 2 == 0:
        # batch.add_user_message(message)
        # summary.set_last_user_input(message)
        entity.set_last_user_input(message)
    else:
        # batch.add_agent_message(message)
        # summary.set_last_agent_input_and_save(message)
        entity.set_last_agent_input_and_save(message)


sys.stdout.write("\nReady to save")
# batch.upsert_all_batches()
# summary.upsert_to_db()
entity.upsert_to_db()


# for log_row in conversation_logs:
#     convo = log_row[0]
#     values = log_row[0].split(';')
# 
#     conversation_id = int(values[0])
#     convo_index = conversation_id - 1
#     role = values[1]
#     content = values[3]
# 
#     if conversation_id != 4:
#         continue
# 
#     sys.stdout.write("\nConversation ID: %s" % conversation_id)
# 
#     if role == 'User':
#         sys.stdout.write("\nAdding user message")
#         summary.set_last_user_input(content)
#         # conversations_entities[convo_index].set_last_user_input(content)
# 
#     if role == 'Agent':
#         sys.stdout.write("\nAdding agent message")
#         summary.set_last_agent_input_and_save(content)
#         # conversations_entities[convo_index].set_last_agent_input_and_save(content)

# conversations_entities[3].upsert_to_db()
#
# for conversation in conversations_summaries:
#     conversation.upsert_to_db()
#     break
#
# for conversation in conversations_entities:
#     conversation.upsert_to_db()
#     break

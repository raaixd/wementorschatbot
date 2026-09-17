import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.knowledge import load_entries
from app.retrieval import Retriever
from app.conversation import ConversationEngine

entries = load_entries()
retriever = Retriever(entries)
engine = ConversationEngine(entries)

queries = [
    'information',
    'information information',
    'Information',
    'INFORMATION',
    'info',
    'more info',
    'give me information',
    'need information',
    'what information',
    'what information do you have',
    'information about classes',
    'information please',
    'tell me information',
    'can i get information',
    'i want information'
]

for q in queries:
    res = retriever.search(q, top_k=3)
    eng_res = engine.handle_message('test-sess', q)
    matches = [f'{item.entry.id} ({item.score:.3f})' for item in res]
    print(f'Query: {q!r}')
    print(f'  Engine intent: {eng_res.intent}')
    print(f'  Engine matched_ids: {eng_res.matched_entry_ids}')
    print(f'  Retriever top matches: {matches}')
    print(f'  Reply: {eng_res.reply[:100]}...')
    if "I can't confirm or invent details" in eng_res.reply:
        print(f'  *** MATCHED POLICY RESTRICTIONS! ***')
    print('-' * 50)

# Open Questions

Facts only the WeMentors team can supply. Until they are answered, the bot says it has no confirmed information.

## Conflicts between the website and the knowledge base

| # | Question | Detail |
|---|---|---|
| 1 | Is **Social Studies** taught, and for which grades? | The demo form and some bot replies list it. The course cards and `middle-school-subjects` do not. |
| 2 | Are **State Boards** supported? | The knowledge base says yes. The website FAQ lists only CBSE, ICSE, IGCSE and IB. |
| 3 | Is the **24-hour** reply promise accurate? | The website and `demo-response-time` promise it. |

## Sensitive topics (answered by fixed replies today)

| # | Question |
|---|---|
| 4 | What support exists for children with additional learning, developmental or physical needs? |
| 5 | How are mentors vetted? What child-safety procedures apply? |
| 6 | How is personal data stored, used and shared? |
| 7 | Please verify the crisis helplines in `backend/app/policy.py` (112 and Tele-MANAS) against official sources. |

## Common questions with no published answer

| # | Question |
|---|---|
| 8 | What are the class timings? Are weekends or evenings available? |
| 9 | How much notice does a demo need, and which slots are offered? |
| 10 | Are classes recorded? Is homework given? Are certificates awarded? |
| 11 | How long is each course, and how can parents pay? |
| 12 | Can a parent attend the demo? Is any software needed? |

## Adding an answer

1. Edit `knowledge/wementors_kb.json`.
2. Regenerate embeddings: `python scripts/generate_kb_embeddings.py` (needs `GEMINI_API_KEY` in the environment).
3. Run `pytest`.

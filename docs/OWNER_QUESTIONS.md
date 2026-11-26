# Questions only the WeMentors team can answer

The bot is built to say "I don't know" instead of guessing, so these gaps stay open until someone with authority answers them. Fill in the **Answer** lines, then add the answers to `knowledge/wementors_kb.json` (see the end of this file). Leave a line blank if the honest answer is "we don't say this publicly".

Source: [KNOWLEDGE_GAPS.md](KNOWLEDGE_GAPS.md) (GAP-01 to GAP-09).

## High priority

**GAP-02. Demo class scheduling**
1. Minimum notice for booking a demo (same day, 24 hours, other)?
   **Answer:**
2. Are demos available on weekends and evenings? Which hours?
   **Answer:**

**GAP-03. Grades 11 and 12 (JEE / NEET / boards)**
3. Is there, or will there be, any support for Grades 11-12? If planned, when?
   **Answer:**

**GAP-01. Fees**
4. Keep the current "fees are personalised, book a demo" answer, or publish a starting price? If a price, what is it and what does it include?
   **Answer:**

## Medium priority

**GAP-04. State boards**
5. Which state boards (for example Maharashtra, Karnataka) are covered, if any?
   **Answer:**

**GAP-06. Refunds, rescheduling and missed classes**
6. What is the refund policy?
   **Answer:**
7. How far in advance can a class be rescheduled, and what happens to a missed class?
   **Answer:**

## Lower priority

**GAP-08. Mentors**
8. What can be said publicly about mentor qualifications and how mentors are selected?
   **Answer:**

**GAP-09. Time zones (NRI / international students)**
9. Which time zones can be supported, and is there a limit on slot times?
   **Answer:**

**GAP-07. Scholarships and sibling discounts**
10. Do any exist? If yes, the exact terms. If not, may the bot say "none are currently offered"?
    **Answer:**

## After answers are in

1. Edit only `knowledge/wementors_kb.json`, adding or updating entries and marking the new facts as verified.
2. Run `python scripts/sync_duplicates.py` to update the other copies (CI checks this).
3. Regenerate embeddings with `python scripts/generate_kb_embeddings.py` (needs a Gemini key).
4. Add one eval case per answered question to `evaluation/questions/dataset.json`, then run the benchmark.
5. Real visitor complaints: export thumbs-down ratings with `python scripts/export_feedback_cases.py --db <chatbot.db>`, label the candidates, and add them to the dataset the same way.

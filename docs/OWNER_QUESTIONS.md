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

## Conflicts to resolve (found 2026-11-26)

The website, the knowledge base and the bot's built-in text disagree on these. Knowledge base edits also need new embeddings, so these are best answered together.

**C-01. Is IELTS its own course?** The website and `programs-overview` list IELTS Preparation as a separate course. `confident-speaker-format` and `confident-speaker-scope` call it part of Confident Speaker.
   **Answer:**

**C-02. Program name.** The website says "Communicative English". Many knowledge base entries and bot replies still say "Confident Speaker". Is the old name retired?
   **Answer:**

**C-03. Communicative English class size.** `confident-speaker-format` says one-to-one. `program-confident-speaker` says small batches. The general answer is 1-on-1 or up to 8 students. What applies to this course?
   **Answer:**

**C-04. Social Studies.** The website's demo form and some bot replies list it. `middle-school-subjects` and the website's course cards do not name it. Is it taught, and for which grades?
   **Answer:**

**C-05. State Boards.** `curricula-supported` includes State Board. The website FAQ lists only CBSE, ICSE, IGCSE and IB. Which is right?
   **Answer:**

**C-06. 24-hour reply.** The website and `demo-response-time` promise contact within 24 hours. The 2026-10-16 answers gave no timeframe. Is the 24-hour promise still accurate?
   **Answer:**

**C-07. Changing mentor or timing.** The website FAQ says families can ask to change mentor, batch or timing. The knowledge base has no entry, so the bot says it does not know. May this be added?
   **Answer:**

## After answers are in

1. Edit only `knowledge/wementors_kb.json`, adding or updating entries and marking the new facts as verified.
2. Run `python scripts/sync_duplicates.py` to update the other copies (CI checks this).
3. Regenerate embeddings with `python scripts/generate_kb_embeddings.py` (needs a Gemini key).
4. Add one eval case per answered question to `evaluation/questions/dataset.json`, then run the benchmark.
5. Real visitor complaints: export thumbs-down ratings with `python scripts/export_feedback_cases.py --db <chatbot.db>`, label the candidates, and add them to the dataset the same way.

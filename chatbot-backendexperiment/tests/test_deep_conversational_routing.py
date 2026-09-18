"""
Comprehensive Regression & Validation Suite for Deep Conversational Intent Routing.

Validates the 42-case semantic intent matrix from Section 22 of the specification:
1. ELIGIBILITY / AUDIENCE (Cases 1-10)
   - "Can I join without being in school?"
   - "Can I join if I'm an adult?"
   - "Can adults join?"
   - "I'm not a student, can I join?"
   - "Do I have to be in school?"
   - "Can someone who isn't in school join?"
   - "I'm a professional, can I join?"
   - "I'm a homemaker, can I join?"
   - "Is WeMentors only for school students?"
   - "I'm not currently studying, can I join?"

2. ACADEMIC PROGRAM DISTINCTION (Cases 11-15)
   - "Tell me about Middle School"
   - "What subjects are in Middle School?"
   - "What does Grade 7 Maths cover?"
   - "Tell me about Foundation Years"
   - "Who is Confident Speaker for?"

3. MIXED-INTENT MESSAGES (Cases 16-20)
   - "I'm an adult and I want to improve my English"
   - "I'm not in school but I want to improve my public speaking"
   - "Can adults join and is it online?"
   - "I'm not a student. What can I learn with WeMentors?"
   - "Can I join without being in school? Also are classes online?"

4. LOCATION / ABOUT WEMENTORS (Cases 21-24)
   - "Where are you located?"
   - "Where is WeMentors based?"
   - "What's your location?"
   - "Where is your office?"

5. DEMO CAPABILITY & INFORMATION (Cases 25-28)
   - "How do I book a demo?"
   - "I want to book a demo"
   - "Can you book a demo for me?"
   - "Book me a demo"

6. CONTEXT OVERRIDE & STALE TOPIC PREVENTION (Cases 29-32)
   - Prior: Middle School -> "Actually, can adults join?"
   - Prior: Demo -> "Where are you located?"
   - Prior: "Can I join?" -> Follow-up: "I'm an adult."
   - Prior: Non-school -> Follow-up: "What about online classes?"

7. NAME DETECTION & CONVERSATIONAL SIGNALS (Cases 33-42)
   - "My name is Raaid"
   - "I'm Raaid"
   - "Raaid"
   - "Never mind"
   - "nevermind"
   - "nvm"
   - "okay"
   - "got it"
   - "Never mind, tell me about Middle School"
   - "Actually, can adults join?"

Guarantees:
- DEMO_TRANSACTION_ENABLED remains False.
- No fake bookings or leads created.
- No stale context hijacking.
- Pure whole-utterance semantic understanding.
"""

import os
import sys
import tempfile
import uuid

# Ensure backend directory is in path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

temp_db = os.path.join(tempfile.mkdtemp(), "test_deep_routing.db")
os.environ["DATABASE_PATH"] = temp_db
os.environ["SKIP_DOTENV"] = "1"

from app import config, database, leads, personality, conversation
from app.knowledge import load_entries

passed_checks = 0
failed_checks = []


def check(description: str, condition: bool, detail: str = ""):
    global passed_checks
    if condition:
        passed_checks += 1
        print(f"[PASS] {description}")
    else:
        failed_checks.append(description)
        print(f"[FAIL] {description} -- {detail}")


def run_tests():
    database.init_db()
    entries = load_entries()
    engine = conversation.ConversationEngine(entries)

    print("\n=======================================================")
    print("CATEGORY 1: ELIGIBILITY / AUDIENCE (10 Cases)")
    print("=======================================================")

    # Case 1: "Can I join without being in school?"
    s1 = f"s1-{uuid.uuid4()}"
    database.ensure_session(s1)
    r1 = engine.handle_message(s1, "Can I join without being in school?")
    check("1. Intent is eligibility", r1.intent.startswith("eligibility"), f"intent={r1.intent}")
    check("1. Does NOT route to Middle School", "middle" not in r1.intent and "Middle School covers Grades 6–8" not in r1.reply, r1.reply[:100])
    check("1. Mentions Confident Speaker for non-school/adult learners", "Confident Speaker" in r1.reply and "Spoken English" in r1.reply, r1.reply[:100])
    check("1. No lead created", database.get_demo_lead(s1) is None)

    # Case 2: "Can I join if I'm an adult?"
    s2 = f"s2-{uuid.uuid4()}"
    database.ensure_session(s2)
    r2 = engine.handle_message(s2, "Can I join if I'm an adult?")
    check("2. Intent is eligibility_adult", r2.intent == "eligibility_adult", f"intent={r2.intent}")
    check("2. Answers eligibility directly before follow-up", "Yes" in r2.reply and "Confident Speaker" in r2.reply, r2.reply[:100])
    check("2. Connects to Confident Speaker (professionals, homemakers)", "professionals" in r2.reply and "homemakers" in r2.reply, r2.reply[:100])
    check("2. Not generic fallback menu", "I can help with WeMentors’ academic programs (Grades 3–10)" not in r2.reply, r2.reply[:100])

    # Case 3: "Can adults join?"
    s3 = f"s3-{uuid.uuid4()}"
    database.ensure_session(s3)
    r3 = engine.handle_message(s3, "Can adults join?")
    check("3. Intent is eligibility_adult", r3.intent == "eligibility_adult", f"intent={r3.intent}")
    check("3. Answers positively and introduces Confident Speaker", "Yes" in r3.reply and "Confident Speaker" in r3.reply, r3.reply[:100])

    # Case 4: "I'm not a student, can I join?"
    s4 = f"s4-{uuid.uuid4()}"
    database.ensure_session(s4)
    r4 = engine.handle_message(s4, "I'm not a student, can I join?")
    check("4. Intent is eligibility_non_school", r4.intent == "eligibility_non_school", f"intent={r4.intent}")
    check("4. Addresses non-student status cleanly", "Confident Speaker" in r4.reply, r4.reply[:100])

    # Case 5: "Do I have to be in school?"
    s5 = f"s5-{uuid.uuid4()}"
    database.ensure_session(s5)
    r5 = engine.handle_message(s5, "Do I have to be in school?")
    check("5. Intent is eligibility_non_school", r5.intent == "eligibility_non_school", f"intent={r5.intent}")
    check("5. Clarifies curriculum vs Confident Speaker", "Grades 3–10" in r5.reply and "Confident Speaker" in r5.reply, r5.reply[:100])

    # Case 6: "Can someone who isn't in school join?"
    s6 = f"s6-{uuid.uuid4()}"
    database.ensure_session(s6)
    r6 = engine.handle_message(s6, "Can someone who isn't in school join?")
    check("6. Intent is eligibility_non_school", r6.intent == "eligibility_non_school", f"intent={r6.intent}")

    # Case 7: "I'm a professional, can I join?"
    s7 = f"s7-{uuid.uuid4()}"
    database.ensure_session(s7)
    r7 = engine.handle_message(s7, "I'm a professional, can I join?")
    check("7. Intent is eligibility_adult", r7.intent == "eligibility_adult", f"intent={r7.intent}")
    check("7. Mentions professionals welcome", "professionals" in r7.reply, r7.reply[:100])

    # Case 8: "I'm a homemaker, can I join?"
    s8 = f"s8-{uuid.uuid4()}"
    database.ensure_session(s8)
    r8 = engine.handle_message(s8, "I'm a homemaker, can I join?")
    check("8. Intent is eligibility_adult", r8.intent == "eligibility_adult", f"intent={r8.intent}")
    check("8. Mentions homemakers welcome", "homemakers" in r8.reply, r8.reply[:100])

    # Case 9: "Is WeMentors only for school students?"
    s9 = f"s9-{uuid.uuid4()}"
    database.ensure_session(s9)
    r9 = engine.handle_message(s9, "Is WeMentors only for school students?")
    check("9. Intent is eligibility_non_school", r9.intent == "eligibility_non_school", f"intent={r9.intent}")

    # Case 10: "I'm not currently studying, can I join?"
    s10 = f"s10-{uuid.uuid4()}"
    database.ensure_session(s10)
    r10 = engine.handle_message(s10, "I'm not currently studying, can I join?")
    check("10. Intent is eligibility_non_school", r10.intent == "eligibility_non_school", f"intent={r10.intent}")

    print("\n=======================================================")
    print("CATEGORY 2: ACADEMIC PROGRAM DISTINCTION (5 Cases)")
    print("=======================================================")

    # Case 11: "Tell me about Middle School"
    s11 = f"s11-{uuid.uuid4()}"
    database.ensure_session(s11)
    r11 = engine.handle_message(s11, "Tell me about Middle School")
    check("11. Intent is middle_school_overview", r11.intent == "middle_school_overview", f"intent={r11.intent}")
    check("11. Mentions Grades 6–8", "Grades 6–8" in r11.reply or "6–8" in r11.reply, r11.reply[:100])

    # Case 12: "What subjects are in Middle School?"
    s12 = f"s12-{uuid.uuid4()}"
    database.ensure_session(s12)
    r12 = engine.handle_message(s12, "What subjects are in Middle School?")
    check("12. Intent is middle_school_subjects", r12.intent == "middle_school_subjects", f"intent={r12.intent}")
    check("12. Mentions core subjects", "Mathematics" in r12.reply and "Science" in r12.reply, r12.reply[:100])

    # Case 13: "What does Grade 7 Maths cover?"
    s13 = f"s13-{uuid.uuid4()}"
    database.ensure_session(s13)
    r13 = engine.handle_message(s13, "What does Grade 7 Maths cover?")
    check("13. Mentions Maths/Middle School context", r13.intent in ("middle_school_subjects", "middle_school_grades", "subject/topic", "faq"), f"intent={r13.intent}")
    check("13. Content covers core Middle School maths", "Mathematics" in r13.reply or "Maths" in r13.reply or "Middle School" in r13.reply, r13.reply[:100])

    # Case 14: "Tell me about Foundation Years"
    s14 = f"s14-{uuid.uuid4()}"
    database.ensure_session(s14)
    r14 = engine.handle_message(s14, "Tell me about Foundation Years")
    check("14. Intent is foundation_years_overview", r14.intent == "foundation_years_overview", f"intent={r14.intent}")
    check("14. Mentions Grades 3–5", "Grades 3–5" in r14.reply, r14.reply[:100])

    # Case 15: "Who is Confident Speaker for?"
    s15 = f"s15-{uuid.uuid4()}"
    database.ensure_session(s15)
    r15 = engine.handle_message(s15, "Who is Confident Speaker for?")
    check("15. Intent is confident_speaker_audience", r15.intent == "confident_speaker_audience", f"intent={r15.intent}")
    check("15. Mentions students, professionals, and homemakers", "professionals" in r15.reply and "homemakers" in r15.reply, r15.reply[:100])

    print("\n=======================================================")
    print("CATEGORY 3: MIXED-INTENT MESSAGES (5 Cases)")
    print("=======================================================")

    # Case 16: "I'm an adult and I want to improve my English"
    s16 = f"s16-{uuid.uuid4()}"
    database.ensure_session(s16)
    r16 = engine.handle_message(s16, "I'm an adult and I want to improve my English")
    check("16. Intent is eligibility_mixed_english", r16.intent == "eligibility_mixed_english", f"intent={r16.intent}")
    check("16. Sets active program to Confident Speaker", database.get_conversation_memory(s16).active_program == "confident_speaker")
    check("16. Directly highlights Confident Speaker", "Confident Speaker" in r16.reply and "Spoken English" in r16.reply, r16.reply[:100])

    # Case 17: "I'm not in school but I want to improve my public speaking"
    s17 = f"s17-{uuid.uuid4()}"
    database.ensure_session(s17)
    r17 = engine.handle_message(s17, "I'm not in school but I want to improve my public speaking")
    check("17. Intent is eligibility_mixed_english", r17.intent == "eligibility_mixed_english", f"intent={r17.intent}")
    check("17. Mentions Public Speaking and Confident Speaker", "Public Speaking" in r17.reply and "Confident Speaker" in r17.reply, r17.reply[:100])

    # Case 18: "Can adults join and is it online?"
    s18 = f"s18-{uuid.uuid4()}"
    database.ensure_session(s18)
    r18 = engine.handle_message(s18, "Can adults join and is it online?")
    check("18. Intent is eligibility_mixed_online", r18.intent == "eligibility_mixed_online", f"intent={r18.intent}")
    check("18. Confirms adult eligibility AND online delivery", "Confident Speaker" in r18.reply and "online" in r18.reply.lower(), r18.reply[:100])

    # Case 19: "I'm not a student. What can I learn with WeMentors?"
    s19 = f"s19-{uuid.uuid4()}"
    database.ensure_session(s19)
    r19 = engine.handle_message(s19, "I'm not a student. What can I learn with WeMentors?")
    check("19. Intent is eligibility_mixed_programs", r19.intent == "eligibility_mixed_programs", f"intent={r19.intent}")
    check("19. Explains Confident Speaker while contrasting academic subjects", "Confident Speaker" in r19.reply and "Grades 3–10" in r19.reply, r19.reply[:100])

    # Case 20: "Can I join without being in school? Also are classes online?"
    s20 = f"s20-{uuid.uuid4()}"
    database.ensure_session(s20)
    r20 = engine.handle_message(s20, "Can I join without being in school? Also are classes online?")
    check("20. Intent is eligibility_mixed_online", r20.intent == "eligibility_mixed_online", f"intent={r20.intent}")
    check("20. Mentions online classes and Confident Speaker", "online" in r20.reply.lower() and "Confident Speaker" in r20.reply, r20.reply[:100])

    print("\n=======================================================")
    print("CATEGORY 4: LOCATION / ABOUT WEMENTORS (4 Cases)")
    print("=======================================================")

    # Case 21: "Where are you located?"
    s21 = f"s21-{uuid.uuid4()}"
    database.ensure_session(s21)
    r21 = engine.handle_message(s21, "Where are you located?")
    check("21. Intent is location", r21.intent == "location", f"intent={r21.intent}")
    check("21. Explains no physical center, classes live and online", "does not operate physical" in r21.reply and "live and online" in r21.reply, r21.reply[:100])
    check("21. Provides official contact email/phone without demo push", "admin@wementors.co" in r21.reply and "+91 76111 92227" in r21.reply, r21.reply[:100])
    check("21. No demo booking instructions", "Click Book Free Demo" not in r21.reply, r21.reply[:100])

    # Case 22: "Where is WeMentors based?"
    s22 = f"s22-{uuid.uuid4()}"
    database.ensure_session(s22)
    r22 = engine.handle_message(s22, "Where is WeMentors based?")
    check("22. Intent is location", r22.intent == "location", f"intent={r22.intent}")

    # Case 23: "What's your location?"
    s23 = f"s23-{uuid.uuid4()}"
    database.ensure_session(s23)
    r23 = engine.handle_message(s23, "What's your location?")
    check("23. Intent is location", r23.intent == "location", f"intent={r23.intent}")

    # Case 24: "Where is your office?"
    s24 = f"s24-{uuid.uuid4()}"
    database.ensure_session(s24)
    r24 = engine.handle_message(s24, "Where is your office?")
    check("24. Intent is location", r24.intent == "location", f"intent={r24.intent}")

    print("\n=======================================================")
    print("CATEGORY 5: DEMO CAPABILITY & INFORMATION (4 Cases)")
    print("=======================================================")

    # Case 25: "How do I book a demo?"
    s25 = f"s25-{uuid.uuid4()}"
    database.ensure_session(s25)
    r25 = engine.handle_message(s25, "How do I book a demo?")
    check("25. Intent is demo_booking", r25.intent == "demo_booking", f"intent={r25.intent}")
    check("25. Directs to Book Free Demo / official channels", "Book Free Demo" in r25.reply or "admin@wementors.co" in r25.reply, r25.reply[:100])
    check("25. DEMO_TRANSACTION_ENABLED is False", leads.DEMO_TRANSACTION_ENABLED is False)
    check("25. No lead created", database.get_demo_lead(s25) is None)

    # Case 26: "I want to book a demo"
    s26 = f"s26-{uuid.uuid4()}"
    database.ensure_session(s26)
    r26 = engine.handle_message(s26, "I want to book a demo")
    check("26. Intent is demo_booking", r26.intent == "demo_booking", f"intent={r26.intent}")
    check("26. Does not ask for user credentials in chat", "what is your name" not in r26.reply.lower() and "your phone" not in r26.reply.lower(), r26.reply[:100])
    check("26. No lead created", database.get_demo_lead(s26) is None)

    # Case 27: "Can you book a demo for me?"
    s27 = f"s27-{uuid.uuid4()}"
    database.ensure_session(s27)
    r27 = engine.handle_message(s27, "Can you book a demo for me?")
    check("27. Intent is demo_transaction_request", r27.intent == "demo_transaction_request", f"intent={r27.intent}")
    check("27. Explains chatbot cannot directly submit bookings", "cannot directly book" in r27.reply.lower() or "direct you to" in r27.reply.lower() or "can't submit" in r27.reply.lower() or "use the book free demo" in r27.reply.lower(), r27.reply[:100])
    check("27. No lead created", database.get_demo_lead(s27) is None)

    # Case 28: "Book me a demo"
    s28 = f"s28-{uuid.uuid4()}"
    database.ensure_session(s28)
    r28 = engine.handle_message(s28, "Book me a demo")
    check("28. Intent is demo_transaction_request", r28.intent == "demo_transaction_request", f"intent={r28.intent}")
    check("28. Explains capability limitation honestly", "cannot directly" in r28.reply.lower() or "direct you" in r28.reply.lower() or "can't submit" in r28.reply.lower() or "book free demo" in r28.reply.lower(), r28.reply[:100])
    check("28. No lead created", database.get_demo_lead(s28) is None)

    print("\n=======================================================")
    print("CATEGORY 6: CONTEXT OVERRIDE & STALE TOPIC (4 Cases)")
    print("=======================================================")

    # Case 29: Turn 1: Middle School -> Turn 2: "Actually, can adults join?"
    s29 = f"s29-{uuid.uuid4()}"
    database.ensure_session(s29)
    engine.handle_message(s29, "Tell me about Middle School")
    mem29 = database.get_conversation_memory(s29)
    check("29a. Prior active program is middle", mem29.active_program == "middle")
    r29b = engine.handle_message(s29, "Actually, can adults join?")
    check("29b. Intent is eligibility_adult (NOT hijacked by Middle School)", r29b.intent == "eligibility_adult", f"intent={r29b.intent}")
    check("29b. Response addresses adult eligibility", "Confident Speaker" in r29b.reply and "Middle School covers Grades 6–8" not in r29b.reply, r29b.reply[:100])

    # Case 30: Turn 1: Demo -> Turn 2: "Where are you located?"
    s30 = f"s30-{uuid.uuid4()}"
    database.ensure_session(s30)
    engine.handle_message(s30, "How does the free demo work?")
    r30b = engine.handle_message(s30, "Where are you located?")
    check("30b. Intent is location (NOT hijacked by demo)", r30b.intent == "location", f"intent={r30b.intent}")
    check("30b. Answers location, not how to book a demo", "does not operate physical" in r30b.reply, r30b.reply[:100])

    # Case 31: Turn 1: "Can I join?" -> Turn 2: "I'm an adult."
    s31 = f"s31-{uuid.uuid4()}"
    database.ensure_session(s31)
    engine.handle_message(s31, "Can I join?")
    r31b = engine.handle_message(s31, "I'm an adult.")
    check("31b. Intent is eligibility_adult", r31b.intent == "eligibility_adult", f"intent={r31b.intent}")
    check("31b. Connects to Confident Speaker", "Confident Speaker" in r31b.reply, r31b.reply[:100])

    # Case 32: Turn 1: "Can I join without being in school?" -> Turn 2: "What about online classes?"
    s32 = f"s32-{uuid.uuid4()}"
    database.ensure_session(s32)
    engine.handle_message(s32, "Can I join without being in school?")
    r32b = engine.handle_message(s32, "What about online classes?")
    check("32b. Intent is online_classes", r32b.intent == "online_classes", f"intent={r32b.intent}")
    check("32b. Answers online class delivery", "live and online" in r32b.reply, r32b.reply[:100])

    print("\n=======================================================")
    print("CATEGORY 7: NAME DETECTION & CONVERSATIONAL (10 Cases)")
    print("=======================================================")

    # Case 33: "My name is Raaid"
    s33 = f"s33-{uuid.uuid4()}"
    database.ensure_session(s33)
    r33 = engine.handle_message(s33, "My name is Raaid")
    check("33. Greets user by name Raaid", "Nice to meet you, Raaid" in r33.reply, r33.reply[:100])

    # Case 34: "I'm Raaid"
    s34 = f"s34-{uuid.uuid4()}"
    database.ensure_session(s34)
    r34 = engine.handle_message(s34, "I'm Raaid")
    check("34. Greets user by name Raaid", "Nice to meet you, Raaid" in r34.reply, r34.reply[:100])

    # Case 35: "Raaid" (standalone without prompt)
    s35 = f"s35-{uuid.uuid4()}"
    database.ensure_session(s35)
    r35 = engine.handle_message(s35, "Raaid")
    check("35. Does NOT fabricate name greeting without assistant having asked", "Nice to meet you, Raaid" not in r35.reply, r35.reply[:100])

    # Case 36: "Never mind"
    s36 = f"s36-{uuid.uuid4()}"
    database.ensure_session(s36)
    r36 = engine.handle_message(s36, "Never mind")
    check("36. Intent is cancellation", r36.intent == "cancellation", f"intent={r36.intent}")
    check("36. NEVER says 'Nice to meet you, Never mind!'", "Nice to meet you" not in r36.reply, r36.reply[:100])

    # Case 37: "nevermind"
    s37 = f"s37-{uuid.uuid4()}"
    database.ensure_session(s37)
    r37 = engine.handle_message(s37, "nevermind")
    check("37. Intent is cancellation", r37.intent == "cancellation", f"intent={r37.intent}")
    check("37. NEVER says 'Nice to meet you, Nevermind!'", "Nice to meet you" not in r37.reply, r37.reply[:100])

    # Case 38: "nvm"
    s38 = f"s38-{uuid.uuid4()}"
    database.ensure_session(s38)
    r38 = engine.handle_message(s38, "nvm")
    check("38. Intent is cancellation", r38.intent == "cancellation", f"intent={r38.intent}")
    check("38. NEVER says 'Nice to meet you, Nvm!'", "Nice to meet you" not in r38.reply, r38.reply[:100])

    # Case 39: "okay"
    s39 = f"s39-{uuid.uuid4()}"
    database.ensure_session(s39)
    r39 = engine.handle_message(s39, "okay")
    check("39. Intent is acknowledgement", r39.intent == "acknowledgement", f"intent={r39.intent}")
    check("39. NEVER treats 'okay' as a name", "Nice to meet you" not in r39.reply, r39.reply[:100])

    # Case 40: "got it"
    s40 = f"s40-{uuid.uuid4()}"
    database.ensure_session(s40)
    r40 = engine.handle_message(s40, "got it")
    check("40. Intent is acknowledgement", r40.intent == "acknowledgement", f"intent={r40.intent}")
    check("40. NEVER treats 'got it' as a name", "Nice to meet you" not in r40.reply, r40.reply[:100])

    # Case 41: "Never mind, tell me about Middle School"
    s41 = f"s41-{uuid.uuid4()}"
    database.ensure_session(s41)
    r41 = engine.handle_message(s41, "Never mind, tell me about Middle School")
    check("41. Intent is middle_school_overview", r41.intent == "middle_school_overview", f"intent={r41.intent}")
    check("41. NEVER greets Never mind as a name", "Nice to meet you" not in r41.reply, r41.reply[:100])
    check("41. Answers Middle School accurately", "Grades 6–8" in r41.reply or "6–8" in r41.reply, r41.reply[:100])

    # Case 42: "Actually, can adults join?"
    s42 = f"s42-{uuid.uuid4()}"
    database.ensure_session(s42)
    r42 = engine.handle_message(s42, "Actually, can adults join?")
    check("42. Intent is eligibility_adult", r42.intent == "eligibility_adult", f"intent={r42.intent}")
    check("42. Strips discourse marker and answers eligibility", "Confident Speaker" in r42.reply and "Nice to meet you" not in r42.reply, r42.reply[:100])

    print("\n=======================================================")
    print(f"RESULTS: Passed {passed_checks} checks, Failed {len(failed_checks)} checks.")
    print("=======================================================")
    if failed_checks:
        print("\nFailed checks:")
        for fc in failed_checks:
            print(f" - {fc}")
        sys.exit(1)
    else:
        print("\nALL 42 DEEP CONVERSATIONAL INTENT ROUTING CASES PASSED PERFECTLY!")


if __name__ == "__main__":
    run_tests()

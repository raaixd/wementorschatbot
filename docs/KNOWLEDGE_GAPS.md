# WeMentors AI Chatbot — Knowledge Gap Analysis & Action Plan

This document catalogues all verified domain knowledge gaps, unlisted policies, and informational ambiguities identified across the WeMentors AI Chatbot knowledge base (`wementors_kb.json`) during baseline evaluation, adversarial stress testing, and system verification.

In adherence to the **Zero Invention Invariant**, the chatbot strictly refuses to fabricate or infer answers for unconfirmed details. Gaps requiring stakeholder verification before knowledge base ingestion are explicitly marked **`[REQUIRES HUMAN INPUT]`**.

---

## 1. Summary of Identified Knowledge Gaps

| Gap ID | Category | Priority | Area / Topic | Status | Impact on User Journey |
|---|---|---|---|---|---|
| **GAP-01** | Commercial | **HIGH** | Exact Tuition Fee Structure & Ranges | Verified Policy / Explicit Fallback | Users asking for exact numbers are informed fees are personalized. |
| **GAP-02** | Operations | **HIGH** | Demo Session Scheduling & Availability Window | **REQUIRES HUMAN INPUT** | Bot directs to web form; cannot confirm same-day/weekend slot lead times. |
| **GAP-03** | Academic | **HIGH** | Grade 11 & 12 (Senior Secondary / JEE / NEET) Support | **REQUIRES HUMAN INPUT** | Current KB only covers Grades 3–10. Inquiries for 11–12 trigger out-of-scope fallback. |
| **GAP-04** | Academic | **MEDIUM** | State Board Curricula Explicit Alignment | **REQUIRES HUMAN INPUT** | CBSE, ICSE, and Cambridge/IB are verified; specific State Boards (e.g. Maharashtra, Karnataka) unverified. |
| **GAP-05** | Academic | **MEDIUM** | Foreign / Regional Language Courses | Verified Negative | French, German, Hindi, Sanskrit inquiries explicitly clarified as unlisted. |
| **GAP-06** | Operations | **MEDIUM** | Refund, Rescheduling, & Missed Class Policy | **REQUIRES HUMAN INPUT** | Bot clarifies policies are unverified and directs to administration. |
| **GAP-07** | Commercial | **LOW** | Scholarships, Sibling Discounts, Financial Aid | Verified Fallback | Unverified; bot directs to official counseling team without inventing discounts. |
| **GAP-08** | Operations | **LOW** | Mentor Qualifications & Hiring Criteria Details | **REQUIRES HUMAN INPUT** | High-level 1-on-1 mentorship verified; formal mentor academic credentials/vetting unlisted. |
| **GAP-09** | Operations | **LOW** | International Time Zone Flexibility for NRI / Global Students | **REQUIRES HUMAN INPUT** | Online global delivery is verified; specific time zone slot commitments require confirmation. |

---

## 2. High-Priority Gaps

### GAP-01: Tuition Fee Pricing & Currency Ranges
- **Description:** Exact tuition fees, monthly retainers, or hourly rates are omitted from public documentation.
- **Questions Impacted:**
  - *"How much does WeMentors cost per month?"*
  - *"What are the fees for Grade 8 Mathematics?"*
  - *"Is there a registration or admission fee?"*
- **Current Behavior:** Returns verified `fees-and-pricing` entry: explains fees are structured individually based on grade band, subject requirements, and session frequency, directing user to consult an advisor after booking a free demo.
- **Recommended Action:** Preserve current policy. If marketing decides to publish transparent starting rates (e.g., *"starting from ₹X,XXX/month"*), this can be ingested after formal approval.
- **Classification:** **`[REQUIRES HUMAN INPUT]`** (Commercial Strategy Decision).

---

### GAP-02: Demo Class Scheduling Lead Time & Slots
- **Description:** The demo duration (30 minutes), format (1-on-1 diagnostic), and platform (live video with digital whiteboard) are verified. However, slot lead times (e.g., minimum 24-hour advance booking, weekend slot availability) are not specified.
- **Questions Impacted:**
  - *"Can I take a demo class today evening?"*
  - *"Are demo classes available on Sundays?"*
- **Current Behavior:** Explains what happens during the demo and points user to the website form and coordinator WhatsApp/email.
- **Recommended Action:** Confirm standard demo booking windows with academic counseling operations and add to `demo-class-available` phrasings/items.
- **Classification:** **`[REQUIRES HUMAN INPUT]`** (Operations).

---

### GAP-03: Senior Secondary (Grades 11–12) & Competitive Exam Guidance
- **Description:** The knowledge base explicit grade band ends at Grade 10 (`program-senior-school`). Parents frequently ask whether WeMentors supports Class 11, Class 12, JEE Foundation, or NEET.
- **Questions Impacted:**
  - *"Do you have coaching for 11th standard Physics?"*
  - *"Can my daughter join for 12th CBSE boards?"*
- **Current Behavior:** Flags Grades 11–12 as unverified/not currently listed, clarifying that academic curriculum programs cover Grades 3–10, and Confident Speaker is open to everyone.
- **Recommended Action:** Formally verify whether WeMentors plans to launch Grade 11–12 academic tracks or remains strictly focused on Grades 3–10.
- **Classification:** **`[REQUIRES HUMAN INPUT]`** (Product Roadmap).

---

## 3. Medium-Priority Gaps

### GAP-04: Explicit State Board Alignment
- **Description:** The KB verifies CBSE, ICSE, Cambridge (IGCSE), and International Baccalaureate (IB). Inquiries regarding specific Indian State Boards (Maharashtra State Board, Karnataka SSLC, Tamil Nadu State Board) rely on semantic similarity to curriculum entries.
- **Questions Impacted:**
  - *"Do you teach according to the Maharashtra State Board syllabus?"*
- **Current Behavior:** Clarifies supported boards and core subjects; does not falsely claim state board accreditation.
- **Recommended Action:** Confirm with academic curriculum heads whether mentors adapt to state board syllabi upon request.
- **Classification:** **`[REQUIRES HUMAN INPUT]`** (Curriculum).

---

### GAP-05: Non-Curriculum Languages & Electives
- **Description:** Verified KB entries document core academic subjects (Mathematics, Science, English, Social Sciences / EVS). Coding, French, German, and Sanskrit are occasionally requested.
- **Current Behavior:** Accurately states that elective coding and foreign languages are not currently in verified offerings; prevents false enrollment expectations.
- **Recommended Action:** Maintain negative constraints; update KB if robotics or language electives are piloted.
- **Classification:** Verified Negative Constraint (No Human Input Required).

---

### GAP-06: Cancellation, Refund, & Session Rescheduling Terms
- **Description:** Cancellation of a demo request is supported in conversation flow. However, paid enrollment terms, missed session rescheduling limits (e.g., 4-hour notice required), and refund policies are absent from KB.
- **Questions Impacted:**
  - *"What happens if my child misses a scheduled session?"*
  - *"Can I get a refund if we discontinue after 2 weeks?"*
- **Current Behavior:** Returns `UNSUPPORTED_DETAILS_FALLBACK` directing user to official administrative contact channels (`admin@wementors.co`, `+91 76111 92227`).
- **Recommended Action:** Provide approved policy summary text for session rescheduling and refund guidelines.
- **Classification:** **`[REQUIRES HUMAN INPUT]`** (Legal / Commercial).

---

## 4. Low-Priority Gaps

### GAP-07: Scholarships, Sibling Concessions, & Corporate Tie-Ups
- **Description:** Financial concessions and merit scholarships are unverified.
- **Current Behavior:** Accurately triggers ambiguous/unsupported fallback directing to counseling staff.
- **Recommended Action:** Document whether WeMentors offers sibling discounts or hardship assistance.
- **Classification:** **`[REQUIRES HUMAN INPUT]`** (Sales Policy).

---

### GAP-08: Mentor Background & Selection Criteria
- **Description:** The KB verifies that each student receives a dedicated 1-on-1 personal mentor (not a rotating roster). Specific details regarding mentor degree qualifications, background checks, and mentor-student pairing algorithms are unlisted.
- **Current Behavior:** Reaffirms 1-on-1 personalized mentorship and weekly parent reporting; avoids hallucinating mentor identities or institutional pedigree.
- **Recommended Action:** Add 2–3 verified factual sentences on mentor selection criteria and training.
- **Classification:** **`[REQUIRES HUMAN INPUT]`** (HR / Quality Assurance).

---

### GAP-09: NRI / Global Time Zone Flexibility
- **Description:** Classes are verified as live, online, and accessible globally. Specific scheduling accommodations for students in North America, Europe, or Southeast Asia (e.g. weekend-only or late-evening IST slots) are not explicitly codified.
- **Current Behavior:** Confirms online delivery from anywhere; directs users to coordinate slot timing via demo form.
- **Recommended Action:** Codify available teaching hours in UTC/EST/GST if NRI expansion is prioritized.
- **Classification:** **`[REQUIRES HUMAN INPUT]`** (Operations).

---

## 5. Ingestion Protocol for New Knowledge

When stakeholder input is provided for any gap above, follow this strict 4-step ingestion protocol:
1. **Draft Entry in Canonical Format:** Question, unique semantic ID, category, list of 10+ realistic phrasings, concise factual answer, 2 relevant follow-ups, and `confidence: "verified"`.
2. **Synchronize All 3 Copies:** Update `knowledge/wementors_kb.json`, `chatbot-backendexperiment/app/wementors_kb.json`, and `api/wementors_kb.json` simultaneously.
3. **Run Regression & Benchmark:** Execute `run_release_test_suite.py` and `run_benchmark.py` to confirm zero regression on existing intents and zero hallucination.
4. **Update Evaluation Baseline:** Record changes in `docs/RAG_EVALUATION.md` and commit atomically.

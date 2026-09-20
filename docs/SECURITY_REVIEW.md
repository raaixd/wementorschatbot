# WeMentors AI Chatbot — Production Security Review & OWASP LLM Audit
**Status:** PRODUCTION READY  
**Classification:** Technical Security Audit  
**Author:** WeMentors Engineering  
**Version:** 1.0  
**Scope:** Frontend Chat Interface, FastAPI Gateway, Conversation Engine, Hybrid Retriever, LLM Failover Pipeline, SQLite Session Store.

---

## 1. Executive Security Summary

The WeMentors AI Chatbot is an informational, consultative assistant designed to guide prospective learners and parents through programs (Grades 3–10 academic support and Confident Speaker), pedagogical philosophy, and enrollment procedures.

By architectural design, the chatbot operates under a **Strict Read-Only & Informational Boundary**:
1. **Zero Booking Agency:** The system cannot book, modify, or cancel trial classes. All bookings must be completed by the user directly on the official website form.
2. **Zero Financial Transactions:** The system has no payment gateway integration, wallet access, or credit card handling.
3. **Deterministic Retrieval Grounding:** Generation is strictly bounded to verified facts in `knowledge/wementors_kb.json`. The assistant never invents unverified fees, schedules, discounts, or academic guarantees.
4. **Resilient Failover:** A 3-tier failover mechanism (`Gemini 3.5 Flash-Lite -> Groq / Llama 3.3 70B -> Deterministic KB Answer`) ensures bounded response delivery with continuous output sanitization even during total API provider outages.

---

## 2. Threat Model Analysis: OWASP Top 10 for LLM Applications (2025)

| OWASP ID | Threat Category | Risk Profile in WeMentors | Implemented Mitigations & Safeguards | Verification Status |
| :--- | :--- | :--- | :--- | :---: |
| **LLM01** | **Prompt Injection** | Moderate (Direct jailbreaks, "DAN", "ignore previous instructions") | Regex-based injection marker detection (`_INJECTION_MARKERS_RE`), immediate deflection to `INJECTION_DEFLECTION` response, LLM system instructions explicitly barring persona override. | **VERIFIED** |
| **LLM02** | **Sensitive Information Disclosure** | High (System prompt extraction, API key leakage, DB student records) | 1. API keys loaded exclusively via environment variables (`os.getenv`), never echoed in logs or responses.<br>2. Prompt leaking deflection filters queries matching "system prompt", "repeat instructions verbatim".<br>3. Session database isolates conversations by ephemeral UUIDs; no PII or lead details are stored if user posts contact info in chat. | **VERIFIED** |
| **LLM03** | **Supply Chain Vulnerabilities** | Low | Python dependencies pinned in `requirements.txt`. Zero heavy agentic frameworks (no LangChain, CrewAI, AutoGen). Zero remote vector database dependencies. | **VERIFIED** |
| **LLM04** | **Data & Model Poisoning** | Low | Knowledge base is stored in a version-controlled, schema-validated JSON file (`wementors_kb.json`). Loaded read-only into memory at server startup. No runtime dynamic knowledge ingestion from untrusted web inputs. | **VERIFIED** |
| **LLM05** | **Improper Output Handling** | High (XSS via Markdown/HTML injection, script tags) | 1. Frontend DOM uses explicit text rendering and markdown parser with HTML tag escaping.<br>2. Backend output sanitization removes raw HTML `<script>` tags and unsanitized links.<br>3. Disallowed formatting stripped prior to HTTP dispatch. | **VERIFIED** |
| **LLM06** | **Excessive Agency** | Critical (False booking claims, fake confirmations) | 1. `DEMO_TRANSACTION_ENABLED = False` enforced as a constant invariant.<br>2. Post-generation action regexes scrub phrases like "I have booked your demo" or "I have noted your name".<br>3. Transactional intent classifier forces guidance to the official top-right website button. | **VERIFIED** |
| **LLM07** | **System Prompt Leakage** | Moderate (Reversing proprietary instructions) | 1. Guardrail checks intercept queries requesting instructions, system prompts, or full internal knowledge bases.<br>2. Deflection returns polite boundary reminder without executing LLM prompt. | **VERIFIED** |
| **LLM08** | **Vector & Embedding Weaknesses** | Low (Vector collision, embedding inversion) | The retrieval engine is a standard-library TF-IDF lexical matcher with exact keyword boosting. There is no vector database or black-box embedding model susceptible to adversarial perturbation. | **VERIFIED** |
| **LLM09** | **Misinformation & Overreliance** | High (Hallucinated fees, false 100% pass guarantees) | 1. KB entries carry explicit `confidence: "verified"` tags.<br>2. Output quality validator strips unsupported guarantees ("100% marks", "guaranteed admission to Oxford").<br>3. Mandatory team referral fallback for unverified queries (`admin@wementors.co` / `+91 76111 92227`). | **VERIFIED** |
| **LLM10** | **Unbounded Consumption / DoS** | High (API token exhaustion, compute starvation) | 1. In-memory IP-based sliding window rate limiter (max 200 req/window).<br>2. Input length capped (max 4,000 characters; longer rejected).<br>3. LLM timeout bounded (10.0s hard timeout with async cancellation).<br>4. Fast fallback prevents hanging connection pools. | **VERIFIED** |

---

## 3. Defense-in-Depth Implementation Details

### 3.1 Input Sanitization Pipeline
Every inbound request to `/api/chat` passes through sequential validation stages:
```
[User HTTP Request]
       │
       ▼
1. JSON Schema & Type Validation (Pydantic / Dict validation)
       │
       ▼
2. Message Length Check (Len <= 4000 characters; non-empty, non-whitespace)
       │
       ▼
3. Rate Limiter Inspection (Sliding window token counter by client IP)
       │
       ▼
4. Adversarial Injection Detection (_INJECTION_MARKERS_RE)
       │─── If Injection Detected ───► Immediate Deflection (Zero LLM Tokens Spent)
       ▼
5. Session Isolation & Context Retrieval (UUID-based SQLite lookup)
```

### 3.2 Output Quality & Anti-Hallucination Pipeline
Before any response is dispatched to the client, `_run_response_quality_checks` executes:
1. **Fee Boundary Enforcement:** If user did not ask about fees, any accidental fee mentions are scrubbed to prevent unsolicited price anchoring.
2. **Action Claim Scrubbing:** Phrases matching `r"\b(i've\s+noted|i\s+saved|demo\s+has\s+been\s+booked)\b"` are rewritten to direct the user to the website form.
3. **Hyperbolic Claim Neutralization:** Any generated claims of "100% marks", "guaranteed score", or "eliminate all anxiety" are automatically rewritten to "individual progress and guided development".
4. **False Affirmation Cleansing:** Accidental leading "Yes." affirmations on ambiguous queries are removed unless backed by an explicit positive match.

### 3.3 Strict Demo Transaction Boundary
The architecture preserves the strict boundary between conversational discovery and lead transaction:
```python
# app/config.py
DEMO_TRANSACTION_ENABLED = False  # Invariant: Must remain False

# app/conversation.py
if _DEMO_TRANSACTION_RE.search(message):
    return ReplyResult(
        personality.DEMO_TRANSACTION_REQUEST_RESPONSE,
        "demo_transaction_request",
        ["how-to-book-demo", "contact-info"],
        1.0,
    )
```
- The assistant explicitly disclaims transaction agency:
  > *"I cannot submit, confirm, or access bookings directly from the chat. You can submit your details using the **Book Free Demo** form at the top-right of the website, or reach out directly to our team at admin@wementors.co or +91 76111 92227."*
- If a user sends personal information (phone numbers, email, student names), the system never writes these to a lead database or claims they are registered; it politely redirects them to the official form.

---

## 4. API & Infrastructure Security

1. **CORS Restrictions:**
   Configured via `ALLOWED_ORIGINS` in `app/config.py`. In production, wildcard `*` is strictly disabled. Allowed origins default to official domains and verified localhost testing ports.
2. **Database Security:**
   SQLite is accessed with parameterized SQL queries (`?` parameter substitutions), completely preventing SQL injection vulnerabilities.
3. **Sensitive Data Storage:**
   No passwords, payment data, or sensitive student records are stored in the chatbot database. Session tables only persist ephemeral chat transcripts for multi-turn reference resolution.
4. **Error Masking:**
   Internal stack traces and database errors are caught at the FastAPI exception handler layer. Clients receive standard structured error responses (`{"error": "..."}`) without sensitive host paths or environment variables leaked.

---

## 5. Security Verification Suite

All security guarantees are verified by automated test suites:
- `tests/test_natural_language_and_false_booking.py` (135 assertions covering booking refusal, lead isolation, and parameter injection)
- `tests/test_production_reliability.py` (20 specific failure paths including excessive length, Unicode fuzzing, XSS, rate limiting, and prompt leaking)
- `evaluation/runners/run_benchmark.py` Category H (Adversarial inputs, jailbreaks, prompt extractions)

**Audit Sign-off:** System complies with production standards for educational AI conversational agents.

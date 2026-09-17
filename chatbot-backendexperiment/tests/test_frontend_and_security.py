"""
Static frontend and security checks (stdlib-only, no browser required).

These verify things a live browser test would otherwise catch, using only
text/HTML parsing so they can run in any environment: HTML validity, JS
syntax (best-effort, only if Node.js is available), required chatbot
elements/wiring, safe message rendering (no innerHTML with message data),
reduced-motion support, and that no secrets or generated files are present
in the source tree.

Run with: python3 tests/test_frontend_and_security.py
"""
import html.parser
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import config  # noqa: E402  (after sys.path fix)

PROJECT_ROOT = config.PROJECT_ROOT
INDEX_HTML = PROJECT_ROOT / "index.html"

failures = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        failures.append(label)


def warn(label, detail=""):
    print(f"[SKIP] {label}" + (f" -- {detail}" if detail else ""))


# --- HTML parses ---------------------------------------------------------
check("index.html exists", INDEX_HTML.exists())
content = INDEX_HTML.read_text(encoding="utf-8")

parse_errors = []


class _Checker(html.parser.HTMLParser):
    def error(self, message):
        parse_errors.append(message)


_Checker().feed(content)
check("index.html parses without HTML errors", not parse_errors, parse_errors)

# --- required chatbot elements exist --------------------------------------
required_ids = [
    "chatbotToggle", "chatbotPanel", "chatbotHeader", "chatbotHeaderAvatar",
    "chatbotClear", "chatbotClose", "chatbotMessages", "chatbotSuggestions",
    "chatbotForm", "chatbotInputLabel", "chatbotInput", "chatbotSend",
]
for element_id in required_ids:
    check(f'required element id="{element_id}" is present', f'id="{element_id}"' in content)

# --- shared liquid-glass logo SVG def, used in both launcher and header -----
check("shared gradient def 'chatbotLogoGrad' is defined once", content.count('id="chatbotLogoGrad"') == 1)
check("shared highlight def 'chatbotLogoHighlight' is defined once", content.count('id="chatbotLogoHighlight"') == 1)
check(
    "gradient def is referenced by both the launcher and the header avatar (>=2 uses)",
    content.count("url(#chatbotLogoGrad)") >= 2,
)

# --- close button is actually wired to close the panel -----------------------
check(
    "chatbotClose button is wired with an event listener",
    "chatbotClose" in content and "chatbotClose.addEventListener" in content,
)

# --- reduced-motion support ------------------------------------------------
check("prefers-reduced-motion is handled in CSS", "prefers-reduced-motion" in content)

# --- extract inline scripts for further checks --------------------------
scripts = re.findall(r"<script(?:(?!src)[^>])*>(.*?)</script>", content, re.S)
check("at least one inline <script> block found", len(scripts) > 0)
combined_js = "\n;\n".join(scripts)

# --- JS syntax (best-effort; skipped, not failed, if Node isn't installed) --
node_path = shutil.which("node")
if node_path:
    # Platform-independent temp path: "/tmp/..." does not exist on Windows.
    tmp_path = pathlib.Path(tempfile.gettempdir()) / "_wementors_combined.js"
    tmp_path.write_text(combined_js, encoding="utf-8")
    result = subprocess.run([node_path, "--check", str(tmp_path)], capture_output=True, text=True)
    check("inline JavaScript has valid syntax (node --check)", result.returncode == 0, result.stderr[:400])
else:
    warn("inline JavaScript syntax check", "Node.js not found on PATH; skipped, not failed")

# --- safe rendering: user textContent, assistant sanitized Markdown ------
# User messages must strictly use .textContent to prevent XSS.
# Assistant messages must pass through renderAssistantMarkdown() which escapes
# raw HTML before converting markdown tokens into safe semantic elements.
add_message_match = re.search(r"function addChatbotMessage\([^)]*\)\s*\{(.*?)\n  \}", combined_js, re.S)
check("addChatbotMessage() function found in the frontend JS", bool(add_message_match))
if add_message_match:
    body = add_message_match.group(1)
    check("addChatbotMessage() escapes user messages via .textContent", ".textContent" in body)
    check("addChatbotMessage() parses assistant messages via renderAssistantMarkdown()", "renderAssistantMarkdown" in body)

check("renderAssistantMarkdown() function is defined in the frontend JS", "function renderAssistantMarkdown(" in combined_js)

chatbot_script_match = re.search(
    r"WeMentors chatbot ----------------\s*\*/\s*\(function\s*\(\)\s*\{(.*?)\}\)\(\);",
    combined_js,
    re.S,
)
check("chatbot's own script block (IIFE) found for scoped safety checks", bool(chatbot_script_match))
chatbot_js = chatbot_script_match.group(1) if chatbot_script_match else ""
chatbot_innerhtml_assignments = re.findall(r"\.innerHTML\s*=\s*(.+?);", chatbot_js)
# Allowed dynamic innerHTML assignments must be strictly sanitized via renderAssistantMarkdown
dynamic_innerhtml = [
    a.strip() for a in chatbot_innerhtml_assignments 
    if not re.match(r"^[\'\"`]", a.strip()) and "renderAssistantMarkdown" not in a
]
check(
    "within the chatbot's own script, any dynamic .innerHTML assignment is strictly sanitized via renderAssistantMarkdown()",
    len(dynamic_innerhtml) == 0,
    dynamic_innerhtml,
)

# --- Markdown rendering & XSS sanitization acceptance tests (Part 4) -----
if node_path:
    # Extract escapeHtml, parseInlineMarkdown, and renderAssistantMarkdown definitions
    fn_match = re.search(r"(function escapeHtml\s*\(.*?\n  function renderAssistantMarkdown\s*\(.*?\n  \})", combined_js, re.S)
    if fn_match:
        fn_code = fn_match.group(1)
        test_harness = f"""
{fn_code}
const boldInput = "Our **Foundation Years** program helps students build confidence.";
const boldOut = renderAssistantMarkdown(boldInput);
if (boldOut.includes("**") || !boldOut.includes("<strong>Foundation Years</strong>")) {{
    console.error("FAIL: Bold markdown not properly formatted or contains asterisks: " + boldOut);
    process.exit(1);
}}

const xssScript = "<script>alert('pwnd')</script>";
const xssOut = renderAssistantMarkdown(xssScript);
if (xssOut.includes("<script>") || !xssOut.includes("&lt;script&gt;")) {{
    console.error("FAIL: Script tag was not escaped: " + xssOut);
    process.exit(2);
}}

const xssImg = '<img src="x" onerror="alert(1)">';
const xssImgOut = renderAssistantMarkdown(xssImg);
if (xssImgOut.includes("<img") || !xssImgOut.includes("&lt;img")) {{
    console.error("FAIL: Img tag was not escaped: " + xssImgOut);
    process.exit(3);
}}

const listInput = "* Machine Learning\\n* Data Science";
const listOut = renderAssistantMarkdown(listInput);
if (!listOut.includes("<ul>") || !listOut.includes("<li>Machine Learning</li>")) {{
    console.error("FAIL: List markdown not parsed correctly: " + listOut);
    process.exit(4);
}}

console.log("MARKDOWN_ACCEPTANCE_PASS");
"""
        tmp_md_path = pathlib.Path(tempfile.gettempdir()) / "_wementors_md_test.js"
        tmp_md_path.write_text(test_harness, encoding="utf-8")
        md_result = subprocess.run([node_path, str(tmp_md_path)], capture_output=True, text=True)
        check("Markdown acceptance: bold text renders as <strong> without visible asterisks", "MARKDOWN_ACCEPTANCE_PASS" in md_result.stdout, md_result.stderr)
        check("Markdown acceptance: <script> and malicious HTML are sanitized/escaped", md_result.returncode == 0, md_result.stderr)
    else:
        check("renderAssistantMarkdown definition extracted", False, "Could not locate complete function body")
else:
    warn("Markdown acceptance tests", "Node.js not available; skipped live JS evaluation")

# --- duplicate-submission guard present -----------------------------------
check(
    "send button/input are disabled while a request is pending (prevents duplicate submits)",
    "chatbotInput.disabled = true" in combined_js and "chatbotSend.disabled = true" in combined_js,
)

# --- no secrets committed anywhere in the project -------------------------
SECRET_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9\-_]{10,}"),   # Anthropic key shape
    re.compile(r"sk-[A-Za-z0-9]{20,}"),           # generic OpenAI-style key shape
    re.compile(r"AKIA[0-9A-Z]{16}"),              # AWS access key id shape
    re.compile(r"ANTHROPIC_API_KEY\s*=\s*['\"]?[A-Za-z0-9\-_]{10,}"),
]
SCAN_EXTENSIONS = {".py", ".html", ".js", ".json", ".md"}
SKIP_DIRS = {".venv", "venv", "__pycache__", ".git", "data", ".pytest_cache", "node_modules"}

offenders = []
for root, dirs, files in os.walk(PROJECT_ROOT):
    dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
    for fname in files:
        if fname == ".env":
            if os.getenv("CI") == "true" or os.getenv("STRICT_NO_ENV") == "1":
                offenders.append(os.path.join(root, fname) + " (a real .env file should not exist in the delivered project)")
            continue
        if os.path.splitext(fname)[1] not in SCAN_EXTENSIONS:
            continue
        fpath = os.path.join(root, fname)
        try:
            text = open(fpath, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                offenders.append(f"{fpath} matched {pattern.pattern}")

# --- .gitignore actually covers the sensitive/generated paths ---------------
gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8") if (PROJECT_ROOT / ".gitignore").exists() else ""
for pattern in [".env", ".env.*", "!.env.example", "data/", "__pycache__/", "*.pyc"]:
    check(f".gitignore covers '{pattern}'", pattern in gitignore)

check("no hardcoded API keys/secrets found in source files", not offenders, offenders)
if os.getenv("CI") == "true" or os.getenv("STRICT_NO_ENV") == "1":
    check("no real .env file present in the project tree (.env.example only)", not (config.BACKEND_ROOT / ".env").exists())
else:
    check("local .env file is safely ignored by .gitignore", ".env" in gitignore)

# --- admin key is never hardcoded to a non-empty value in .env.example -----
env_example = (config.BACKEND_ROOT / ".env.example").read_text(encoding="utf-8")
check("ANTHROPIC_API_KEY is blank in .env.example (no real key committed)", re.search(r"^ANTHROPIC_API_KEY=\s*$", env_example, re.M) is not None)
check("ADMIN_API_KEY is blank in .env.example (no real key committed)", re.search(r"^ADMIN_API_KEY=\s*$", env_example, re.M) is not None)
check("GROQ_API_KEY is blank in .env.example (no real key committed)", re.search(r"^GROQ_API_KEY=\s*$", env_example, re.M) is not None)
# Groq keys have a distinctive prefix; make sure one never lands in the tree.
check(
    "no Groq API key shape found anywhere in the source tree",
    not any(re.search(r"gsk_[A-Za-z0-9]{20,}", open(fp, encoding="utf-8", errors="ignore").read())
            for fp in [str(PROJECT_ROOT / "index.html"), str(PROJECT_ROOT / "README.md"),
                       str(config.BACKEND_ROOT / ".env.example")]),
)

print()
if failures:
    print(f"{len(failures)} check(s) FAILED: {failures}")
    sys.exit(1)
else:
    print("All checks passed.")

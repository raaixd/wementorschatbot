"""
WeMentors AI Chatbot — Master Report PDF Generator
Parses docs/PROJECT_REPORT.md into a high-fidelity, beautifully styled HTML document
and compiles it to docs/PROJECT_REPORT.pdf using headless Chrome/Edge.
"""
import os
import re
import subprocess
import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
MD_PATH = ROOT_DIR / "docs" / "PROJECT_REPORT.md"
HTML_PATH = ROOT_DIR / "docs" / "project_report.html"
PDF_PATH = ROOT_DIR / "docs" / "PROJECT_REPORT.pdf"
DOCS_ALT_PDF_PATH = ROOT_DIR / "docs" / "WeMentors_Project_Report.pdf"
ROOT_PDF_PATH = ROOT_DIR / "WeMentors_Project_Report.pdf"

def parse_markdown_to_html(md_text: str) -> str:
    lines = md_text.splitlines()
    html_parts = []
    
    in_cover = True
    in_code = False
    code_lines = []
    code_lang = ""
    in_list = False
    list_items = []
    
    current_q_card = []
    in_q_card = False
    
    current_adr_card = []
    in_adr_card = False
    
    current_incident_card = []
    in_incident_card = False

    def inline_format(text: str) -> str:
        text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
        text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
        text = re.sub(r"\[(.+?)\]\((.+?)\)", r'<a href="\2">\1</a>', text)
        text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
        text = re.sub(r"\$([^\$]+)\$", r"<span class='math'>\1</span>", text)
        return text

    def parse_ascii_table(raw_code: str):
        raw_lines = [l.strip() for l in raw_code.strip().splitlines() if l.strip()]
        if not raw_lines:
            return None
        table_lines = [l for l in raw_lines if '|' in l and not l.startswith('+--')]
        if len(table_lines) < 2:
            return None
        
        header_idx = 0
        title = None
        first_row_cols = [c.strip() for c in table_lines[0].split('|') if c.strip()]
        if len(first_row_cols) == 1:
            title = first_row_cols[0].strip()
            header_idx = 1
        
        if header_idx >= len(table_lines):
            return None
            
        headers = [c.strip() for c in table_lines[header_idx].split('|')[1:-1]]
        rows = []
        for l in table_lines[header_idx+1:]:
            cols = [c.strip() for c in l.split('|')[1:-1]]
            if cols and any(cols):
                rows.append(cols)
                
        if not headers or not rows:
            return None
            
        out = ["<div class='table-container'>"]
        if title:
            out.append(f"<div class='table-title'>{inline_format(title)}</div>")
        out.append("<table><thead><tr>")
        for h in headers:
            out.append(f"<th>{inline_format(h)}</th>")
        out.append("</tr></thead><tbody>")
        
        for row in rows:
            out.append("<tr>")
            for i, c in enumerate(row):
                val_fmt = inline_format(c)
                # Apply special cell badges
                if val_fmt in ("PASS", "EXCEEDED", "VERIFIED", "100.00%", "100.0%", "88.10%", "88.1%", "Complete", "Exceeded"):
                    out.append(f"<td><span class='badge-pass'>{val_fmt}</span></td>")
                elif "DO NOT SAY" in headers and i == 0:
                    out.append(f"<td class='cell-warning'>{val_fmt}</td>")
                elif "SAY INSTEAD" in headers and i == 1:
                    out.append(f"<td class='cell-success'>{val_fmt}</td>")
                elif val_fmt in ("NOT WIRED", "NOT MEASURED", "DISABLED", "MODEL 404"):
                    out.append(f"<td><span class='badge-muted'>{val_fmt}</span></td>")
                else:
                    out.append(f"<td>{val_fmt}</td>")
            out.append("</tr>")
        out.append("</tbody></table></div>")
        return "\n".join(out)

    def parse_evolution_block(raw_code: str):
        if "THE MEASURABLE ENGINEERING EVOLUTION" not in raw_code:
            return None
        return """
        <div class="evolution-container">
          <div class="evo-header">The Measurable Engineering Evolution</div>
          <div class="evo-grid">
            <div class="evo-col evo-baseline">
              <div class="evo-col-title">1. Baseline Prototype (Initial State)</div>
              <ul>
                <li>Naive retrieval scoring with raw token overlap</li>
                <li>Single LLM provider dependency (Gemini alone)</li>
                <li>Unguided multi-turn follow-ups & pronoun drift</li>
                <li>Unhandled provider rate limits (429 API crashes)</li>
                <li>Strict string test assertion fragility</li>
              </ul>
              <div class="evo-score-box evo-score-base">
                <span class="evo-score-num">63.5%</span>
                <span class="evo-score-label">Answer Correctness</span>
              </div>
            </div>
            <div class="evo-col evo-production">
              <div class="evo-col-title">2. Production Release v1.0 (Hardened)</div>
              <ul>
                <li>Weighted lexical-semantic retriever with coverage gating</li>
                <li>Bounded 3-tier fallback (Gemini &rarr; Groq &rarr; Safe KB)</li>
                <li>Contextual reference rewriting with fee-intent bypass</li>
                <li>Active Groq model inspection & sub-second failover</li>
                <li>Deterministic output normalization & quality gates</li>
              </ul>
              <div class="evo-score-box evo-score-prod">
                <span class="evo-score-num">88.1%</span>
                <span class="evo-score-label">Answer Correctness (+24.6 pp)</span>
              </div>
            </div>
          </div>
        </div>
        """

    def parse_flowchart_block(raw_code: str):
        if "Incoming User Query" in raw_code and "Session & State" in raw_code:
            return """
            <div class="flowchart-container">
              <div class="fc-step fc-input">Incoming User Query</div>
              <div class="fc-arrow">&darr;</div>
              <div class="fc-step">Session & State Resolution (UUID Tracking & Sliding History)</div>
              <div class="fc-arrow">&darr;</div>
              <div class="fc-step">Contextual Reference Rewriting (Pronoun/Ordinal Resolution & Fee Bypass)</div>
              <div class="fc-arrow">&darr;</div>
              <div class="fc-step">Weighted Lexical-Semantic Retrieval (TF-IDF + Domain Stemming + Coverage Gating)</div>
              <div class="fc-arrow">&darr;</div>
              <div class="fc-split">
                <div class="fc-branch fc-high">
                  <div class="fc-branch-title">Score &ge; 0.12 (High Confidence)</div>
                  <div class="fc-substep"><strong>Bounded 3-Tier LLM Generation:</strong></div>
                  <div class="fc-tier">1. Primary: Gemini 3.5 Flash-Lite (4.0s timeout)</div>
                  <div class="fc-tier">2. Fallback: Groq (GPT-OSS 120B / Qwen 27B)</div>
                  <div class="fc-tier">3. Offline: Deterministic KB Template Answer</div>
                </div>
                <div class="fc-branch fc-low">
                  <div class="fc-branch-title">Score &lt; 0.12 (Low Confidence / Unmatched)</div>
                  <div class="fc-substep"><strong>Deterministic Grounded Fallback:</strong></div>
                  <div class="fc-tier">Honest safe referral directing to admin@wementors.co & +91 76111 92227</div>
                </div>
              </div>
              <div class="fc-arrow">&darr;</div>
              <div class="fc-step">Post-Generation Quality & Safety Gates (Strip False Bookings, Neutralize Guarantees, Normalize Unicode)</div>
              <div class="fc-arrow">&darr;</div>
              <div class="fc-step fc-output">Verified Response & Observability Event (HTTP 200 OK + JSON Telemetry)</div>
            </div>
            """
        return None

    def flush_code():
        nonlocal in_code, code_lines, code_lang
        if not code_lines:
            return ""
        raw_code = "\n".join(code_lines)
        code_lines = []
        in_code = False
        
        # 1. Check if this is the key metrics strip
        if "KEY METRICS STRIP" in raw_code:
            return """
            <div class="metrics-grid">
              <div class="metric-card highlight">
                <div class="metric-num">88.1%</div>
                <div class="metric-label">Answer Correctness</div>
                <div class="metric-sub">126-Question Curated Benchmark</div>
              </div>
              <div class="metric-card">
                <div class="metric-num">+24.6 pp</div>
                <div class="metric-label">vs Baseline Score</div>
                <div class="metric-sub">63.5% &rarr; 88.1% Measured Gain</div>
              </div>
              <div class="metric-card">
                <div class="metric-num">96.83%</div>
                <div class="metric-label">Context Resolution</div>
                <div class="metric-sub">Multi-Turn Pronouns & Ordinals</div>
              </div>
              <div class="metric-card">
                <div class="metric-num">20 / 20</div>
                <div class="metric-label">Release Test Suites</div>
                <div class="metric-sub">1,146+ Automated Assertions</div>
              </div>
            </div>
            <div class="metrics-grid secondary-strip">
              <div class="metric-card-mini">
                <span class="m-val">100.0%</span>
                <span class="m-lbl">Demo Safety (0 Leaks)</span>
              </div>
              <div class="metric-card-mini">
                <span class="m-val">100.0%</span>
                <span class="m-lbl">Groundedness Score</span>
              </div>
              <div class="metric-card-mini">
                <span class="m-val">0.0%</span>
                <span class="m-lbl">Benchmark Hallucination</span>
              </div>
              <div class="metric-card-mini">
                <span class="m-val">41</span>
                <span class="m-lbl">Verified KB Entries</span>
              </div>
            </div>
            """

        # 2. Check for Evolution Block
        evo_html = parse_evolution_block(raw_code)
        if evo_html:
            return evo_html

        # 3. Check for Flowchart Block
        fc_html = parse_flowchart_block(raw_code)
        if fc_html:
            return fc_html

        # 4. Check for ASCII Table
        table_html = parse_ascii_table(raw_code)
        if table_html:
            return table_html

        escaped = raw_code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return f"<pre class='code-block'><code>{escaped}</code></pre>"

    def flush_list():
        nonlocal in_list, list_items
        if not list_items:
            return ""
        out = ["<ul class='styled-list'>"]
        for li in list_items:
            out.append(f"<li>{inline_format(li)}</li>")
        out.append("</ul>")
        list_items = []
        in_list = False
        return "\n".join(out)

    def flush_q_card():
        nonlocal in_q_card, current_q_card
        if not current_q_card:
            return ""
        q_html = render_question_card(current_q_card)
        current_q_card = []
        in_q_card = False
        return q_html

    def flush_adr_card():
        nonlocal in_adr_card, current_adr_card
        if not current_adr_card:
            return ""
        adr_html = render_adr_card(current_adr_card)
        current_adr_card = []
        in_adr_card = False
        return adr_html

    def flush_incident_card():
        nonlocal in_incident_card, current_incident_card
        if not current_incident_card:
            return ""
        inc_html = render_incident_card(current_incident_card)
        current_incident_card = []
        in_incident_card = False
        return inc_html

    def render_question_card(lines_list: list) -> str:
        q_title = lines_list[0].replace("####", "").strip()
        body_lines = lines_list[1:]
        out = [
            f"<div class='question-card'>",
            f"<div class='q-header'><span class='q-badge'>Interview Question</span> <h4>{inline_format(q_title)}</h4></div>",
            "<div class='q-body'>"
        ]
        
        for bline in body_lines:
            bline = bline.strip()
            if not bline:
                continue
            if bline.startswith("- **Short Answer:**"):
                content = bline.replace("- **Short Answer:**", "").strip()
                out.append(f"<div class='qa-row qa-short'><span class='qa-tag tag-short'>Short Interview Answer</span> <p>{inline_format(content)}</p></div>")
            elif bline.startswith("- **Deep Technical Answer:**"):
                content = bline.replace("- **Deep Technical Answer:**", "").strip()
                out.append(f"<div class='qa-row qa-deep'><span class='qa-tag tag-deep'>Deeper Technical Defense</span> <p>{inline_format(content)}</p></div>")
            elif bline.startswith("- **Follow-up Question:**"):
                content = bline.replace("- **Follow-up Question:**", "").strip()
                out.append(f"<div class='qa-row qa-followup-q'><span class='qa-tag tag-followup'>Follow-up Question</span> <p>{inline_format(content)}</p></div>")
            elif bline.startswith("- **Follow-up Answer:**"):
                content = bline.replace("- **Follow-up Answer:**", "").strip()
                out.append(f"<div class='qa-row qa-followup-a'><span class='qa-tag tag-fanswer'>Follow-up Defense</span> <p>{inline_format(content)}</p></div>")
            elif bline.startswith("- **What NOT to Claim:**"):
                content = bline.replace("- **What NOT to Claim:**", "").strip()
                out.append(f"<div class='qa-row qa-guardrail'><span class='qa-tag tag-warning'>What NOT to Claim</span> <p>{inline_format(content)}</p></div>")
            elif bline.startswith("- "):
                out.append(f"<p class='qa-bullet'>&bull; {inline_format(bline[2:])}</p>")
            else:
                out.append(f"<p>{inline_format(bline)}</p>")
        out.append("</div></div>")
        return "\n".join(out)

    def render_adr_card(lines_list: list) -> str:
        title = lines_list[0].replace("###", "").strip()
        body_lines = lines_list[1:]
        out = [
            f"<div class='adr-card'>",
            f"<div class='adr-header'><span class='adr-badge'>Architectural Decision Record</span> <h4>{inline_format(title)}</h4></div>",
            "<div class='adr-body'>"
        ]
        for bline in body_lines:
            bline = bline.strip()
            if not bline:
                continue
            if bline.startswith("- **Decision:**"):
                c = bline.replace("- **Decision:**", "").strip()
                out.append(f"<div class='adr-item'><span class='adr-label'>Decision:</span> <span class='adr-val'>{inline_format(c)}</span></div>")
            elif bline.startswith("- **Why:**"):
                c = bline.replace("- **Why:**", "").strip()
                out.append(f"<div class='adr-item'><span class='adr-label'>Why:</span> <span class='adr-val'>{inline_format(c)}</span></div>")
            elif bline.startswith("- **Trade-off:**"):
                c = bline.replace("- **Trade-off:**", "").strip()
                out.append(f"<div class='adr-item'><span class='adr-label'>Trade-off:</span> <span class='adr-val'>{inline_format(c)}</span></div>")
            elif bline.startswith("- **Alternative:**"):
                c = bline.replace("- **Alternative:**", "").strip()
                out.append(f"<div class='adr-item'><span class='adr-label'>Alternative Considered:</span> <span class='adr-val'>{inline_format(c)}</span></div>")
            elif bline.startswith("- **Why Alternative Not Used:**"):
                c = bline.replace("- **Why Alternative Not Used:**", "").strip()
                out.append(f"<div class='adr-item'><span class='adr-label'>Why Alternative Rejected:</span> <span class='adr-val'>{inline_format(c)}</span></div>")
            else:
                out.append(f"<p>{inline_format(bline)}</p>")
        out.append("</div></div>")
        return "\n".join(out)

    def render_incident_card(lines_list: list) -> str:
        title = lines_list[0].replace("###", "").strip()
        body_lines = lines_list[1:]
        out = [
            f"<div class='incident-card'>",
            f"<div class='inc-header'><span class='inc-badge'>Production Incident Case Study</span> <h4>{inline_format(title)}</h4></div>",
            "<div class='inc-body'>"
        ]
        for bline in body_lines:
            bline = bline.strip()
            if not bline:
                continue
            if bline.startswith("- **Problem:**"):
                c = bline.replace("- **Problem:**", "").strip()
                out.append(f"<div class='inc-step step-problem'><span class='inc-step-tag tag-prob'>1. Problem Observed</span> <p>{inline_format(c)}</p></div>")
            elif bline.startswith("- **Root Cause:**"):
                c = bline.replace("- **Root Cause:**", "").strip()
                out.append(f"<div class='inc-step step-cause'><span class='inc-step-tag tag-cause'>2. Root Cause Analysis</span> <p>{inline_format(c)}</p></div>")
            elif bline.startswith("- **Fix:**"):
                c = bline.replace("- **Fix:**", "").strip()
                out.append(f"<div class='inc-step step-fix'><span class='inc-step-tag tag-fix'>3. Engineering Fix</span> <p>{inline_format(c)}</p></div>")
            elif bline.startswith("- **Result:**"):
                c = bline.replace("- **Result:**", "").strip()
                out.append(f"<div class='inc-step step-res'><span class='inc-step-tag tag-res'>4. Verified Result</span> <p>{inline_format(c)}</p></div>")
            elif bline.startswith("- **Engineering Lesson:**"):
                c = bline.replace("- **Engineering Lesson:**", "").strip()
                out.append(f"<div class='inc-step step-lesson'><span class='inc-step-tag tag-lesson'>5. Engineering Takeaway</span> <p>{inline_format(c)}</p></div>")
            elif re.match(r"^\d+\.\s+", bline):
                out.append(f"<p class='inc-substep'>{inline_format(bline)}</p>")
            else:
                out.append(f"<p>{inline_format(bline)}</p>")
        out.append("</div></div>")
        return "\n".join(out)

    for line in lines:
        if line.startswith("```"):
            if in_code:
                html_parts.append(flush_code())
            else:
                if in_list: html_parts.append(flush_list())
                in_code = True
                code_lines = []
                code_lang = line.replace("```", "").strip()
            continue

        if in_code:
            code_lines.append(line)
            continue

        if line.startswith("#### Question "):
            if in_q_card: html_parts.append(flush_q_card())
            if in_adr_card: html_parts.append(flush_adr_card())
            if in_incident_card: html_parts.append(flush_incident_card())
            if in_list: html_parts.append(flush_list())
            in_q_card = True
            current_q_card = [line]
            continue
        elif in_q_card:
            if line.startswith("### ") or line.startswith("## ") or line.startswith("# ") or line.startswith("#### "):
                html_parts.append(flush_q_card())
            else:
                current_q_card.append(line)
                continue

        if line.startswith("### Decision "):
            if in_q_card: html_parts.append(flush_q_card())
            if in_adr_card: html_parts.append(flush_adr_card())
            if in_incident_card: html_parts.append(flush_incident_card())
            if in_list: html_parts.append(flush_list())
            in_adr_card = True
            current_adr_card = [line]
            continue
        elif in_adr_card:
            if line.startswith("### ") or line.startswith("## ") or line.startswith("# "):
                html_parts.append(flush_adr_card())
            else:
                current_adr_card.append(line)
                continue

        if line.startswith("### Incident "):
            if in_q_card: html_parts.append(flush_q_card())
            if in_adr_card: html_parts.append(flush_adr_card())
            if in_incident_card: html_parts.append(flush_incident_card())
            if in_list: html_parts.append(flush_list())
            in_incident_card = True
            current_incident_card = [line]
            continue
        elif in_incident_card:
            if line.startswith("### ") or line.startswith("## ") or line.startswith("# "):
                html_parts.append(flush_incident_card())
            else:
                current_incident_card.append(line)
                continue

        if line.startswith("- ") or line.startswith("* "):
            if not in_list:
                in_list = True
                list_items = [line[2:]]
            else:
                list_items.append(line[2:])
            continue
        elif in_list:
            html_parts.append(flush_list())

        if line.startswith("# "):
            title = line[2:].strip()
            html_parts.append(f"<div class='cover-header'><span class='brand-pill'>WeMentors AI Engineering</span><h1 class='main-title'>{inline_format(title)}</h1>")
            continue
        elif line.startswith("## "):
            sec_title = line[3:].strip()
            if in_cover and "Production-Oriented" in sec_title:
                html_parts.append(f"<h2 class='cover-subtitle'>{inline_format(sec_title)}</h2>")
                continue
            if in_cover and "Executive Summary" in sec_title:
                in_cover = False
                html_parts.append("</div><!-- end cover -->\n<div class='page-break'></div>\n")
            html_parts.append(f"<h2 class='section-title'>{inline_format(sec_title)}</h2>")
            continue
        elif line.startswith("### "):
            sub_title = line[4:].strip()
            if in_cover and "Engineering, Evaluation" in sub_title:
                html_parts.append(f"<p class='cover-sub-caption'>{inline_format(sub_title)}</p>")
                continue
            html_parts.append(f"<h3 class='sub-section-title'>{inline_format(sub_title)}</h3>")
            continue
        elif line.startswith("#### "):
            h4_title = line[5:].strip()
            html_parts.append(f"<h4 class='h4-title'>{inline_format(h4_title)}</h4>")
            continue

        if line.startswith("> "):
            html_parts.append(f"<blockquote class='styled-quote'>{inline_format(line[2:])}</blockquote>")
            continue

        if line.strip() in ("---", "===", "***"):
            html_parts.append("<hr class='section-divider'>")
            continue

        if not line.strip():
            continue

        html_parts.append(f"<p>{inline_format(line)}</p>")

    if in_code: html_parts.append(flush_code())
    if in_list: html_parts.append(flush_list())
    if in_q_card: html_parts.append(flush_q_card())
    if in_adr_card: html_parts.append(flush_adr_card())
    if in_incident_card: html_parts.append(flush_incident_card())

    return "\n".join(html_parts)


def build_html_document(body_html: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>WeMentors AI Chatbot — Production Release & Interview Defense Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  @page {{
    size: A4;
    margin: 15mm 13mm 15mm 13mm;
    @top-left {{
      content: "WeMentors AI Chatbot \u2022 Production Engineering Case Study";
      font-family: 'Plus Jakarta Sans', sans-serif;
      font-size: 7pt;
      color: #94a3b8;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    @top-right {{
      content: "v1.0 Verified Release";
      font-family: 'JetBrains Mono', monospace;
      font-size: 7pt;
      color: #94a3b8;
    }}
    @bottom-right {{
      content: "Page " counter(page);
      font-family: 'JetBrains Mono', monospace;
      font-size: 7.5pt;
      color: #64748b;
    }}
    @bottom-left {{
      content: "Certified Defensible AI Systems Engineering Report";
      font-family: 'Plus Jakarta Sans', sans-serif;
      font-size: 7pt;
      color: #94a3b8;
    }}
  }}

  *, *::before, *::after {{
    box-sizing: border-box;
  }}

  body {{
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #0f172a;
    background: #ffffff;
    line-height: 1.52;
    font-size: 9.3pt;
    margin: 0;
    padding: 0;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }}

  .page-break {{
    page-break-after: always;
    break-after: page;
  }}

  h1, h2, h3, h4 {{
    color: #0f172a;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-top: 16px;
    margin-bottom: 7px;
  }}

  p {{
    margin: 0 0 8px 0;
    color: #1e293b;
  }}

  a {{
    color: #2563eb;
    text-decoration: none;
  }}

  code {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 8.2pt;
    background: #f1f5f9;
    padding: 1px 4px;
    border-radius: 4px;
    color: #0f172a;
    border: 1px solid #e2e8f0;
  }}

  .math {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 8.5pt;
    color: #334155;
    background: #f8fafc;
    padding: 1px 4px;
    border-radius: 3px;
  }}

  /* Cover Header */
  .cover-header {{
    padding: 18px 0 14px 0;
    border-bottom: 2px solid #0f172a;
    margin-bottom: 20px;
  }}

  .brand-pill {{
    font-size: 7.5pt;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #3730a3;
    background: #e0e7ff;
    padding: 4px 10px;
    border-radius: 9999px;
    display: inline-block;
    margin-bottom: 10px;
  }}

  .main-title {{
    font-size: 24pt;
    font-weight: 800;
    letter-spacing: -0.035em;
    line-height: 1.1;
    margin: 2px 0 4px 0;
    color: #0f172a;
  }}

  .cover-subtitle {{
    font-size: 12.5pt;
    font-weight: 600;
    color: #2563eb;
    margin: 0 0 3px 0;
    letter-spacing: -0.01em;
  }}

  .cover-sub-caption {{
    font-size: 9pt;
    color: #64748b;
    font-weight: 500;
    margin: 0 0 14px 0;
  }}

  /* Metric Strips */
  .metrics-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin: 16px 0;
  }}

  .metric-card {{
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 7px;
    padding: 12px 10px;
    text-align: center;
  }}

  .metric-card.highlight {{
    background: #eff6ff;
    border: 1.5px solid #93c5fd;
  }}

  .metric-num {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 18.5pt;
    font-weight: 800;
    color: #0f172a;
    line-height: 1.1;
  }}

  .metric-card.highlight .metric-num {{
    color: #1d4ed8;
  }}

  .metric-label {{
    font-size: 7.5pt;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #475569;
    margin-top: 4px;
  }}

  .metric-sub {{
    font-size: 7pt;
    color: #64748b;
    margin-top: 2px;
    font-family: 'JetBrains Mono', monospace;
  }}

  .secondary-strip {{
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
    margin-top: 0;
    margin-bottom: 20px;
  }}

  .metric-card-mini {{
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 7px 9px;
    display: flex;
    justify-content: space-between;
    align-items: baseline;
  }}

  .m-val {{
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 10pt;
    color: #0f172a;
  }}

  .m-lbl {{
    font-size: 7pt;
    color: #64748b;
    font-weight: 600;
    text-transform: uppercase;
  }}

  /* Headings */
  .section-title {{
    font-size: 14pt;
    font-weight: 800;
    border-bottom: 1.5px solid #0f172a;
    padding-bottom: 5px;
    margin-top: 24px;
    margin-bottom: 10px;
    page-break-after: avoid;
    break-after: avoid;
  }}

  .sub-section-title {{
    font-size: 11pt;
    font-weight: 700;
    color: #1e293b;
    margin-top: 16px;
    margin-bottom: 7px;
    page-break-after: avoid;
    break-after: avoid;
  }}

  .h4-title {{
    font-size: 10pt;
    font-weight: 700;
    color: #334155;
    margin-top: 10px;
    margin-bottom: 4px;
    page-break-after: avoid;
    break-after: avoid;
  }}

  .section-divider {{
    border: none;
    border-top: 1px dashed #cbd5e1;
    margin: 18px 0;
  }}

  .styled-list {{
    margin: 3px 0 10px 16px;
    padding: 0;
  }}

  .styled-list li {{
    margin-bottom: 3px;
    color: #1e293b;
  }}

  /* Tables */
  .table-container {{
    margin: 12px 0 16px 0;
    page-break-inside: avoid;
    break-inside: avoid;
  }}

  .table-title {{
    font-size: 8pt;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #475569;
    margin-bottom: 4px;
    font-family: 'JetBrains Mono', monospace;
  }}

  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 8.2pt;
    line-height: 1.35;
  }}

  th {{
    background: #f1f5f9;
    color: #0f172a;
    font-weight: 700;
    text-align: left;
    padding: 6px 8px;
    border-top: 1px solid #cbd5e1;
    border-bottom: 2px solid #0f172a;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-size: 7.2pt;
  }}

  td {{
    padding: 5px 8px;
    border-bottom: 1px solid #e2e8f0;
    color: #1e293b;
  }}

  tr:nth-child(even) td {{
    background: #f8fafc;
  }}

  .badge-pass {{
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    color: #15803d;
    background: #dcfce7;
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 7.5pt;
  }}

  .badge-muted {{
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    color: #64748b;
    background: #f1f5f9;
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 7.5pt;
  }}

  .cell-warning {{
    color: #991b1b;
    background: #fef2f2 !important;
    font-weight: 600;
  }}

  .cell-success {{
    color: #166534;
    background: #f0fdf4 !important;
    font-weight: 600;
  }}

  /* Code Blocks */
  pre.code-block {{
    background: #0f172a;
    color: #f8fafc;
    padding: 9px 12px;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 7.8pt;
    line-height: 1.4;
    overflow-x: hidden;
    white-space: pre-wrap;
    margin: 8px 0 12px 0;
    page-break-inside: avoid;
    break-inside: avoid;
  }}

  pre.code-block code {{
    background: transparent;
    color: inherit;
    padding: 0;
    border: none;
  }}

  /* Blockquote */
  blockquote.styled-quote {{
    border-left: 3px solid #3b82f6;
    background: #f8fafc;
    margin: 8px 0;
    padding: 6px 12px;
    font-style: italic;
    color: #334155;
  }}

  /* Evolution Side-by-Side */
  .evolution-container {{
    border: 1px solid #cbd5e1;
    border-radius: 7px;
    background: #ffffff;
    margin: 14px 0;
    page-break-inside: avoid;
    break-inside: avoid;
  }}

  .evo-header {{
    background: #f1f5f9;
    padding: 6px 12px;
    font-weight: 700;
    font-size: 8pt;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #334155;
    border-bottom: 1px solid #cbd5e1;
    font-family: 'JetBrains Mono', monospace;
  }}

  .evo-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0;
  }}

  .evo-col {{
    padding: 12px 14px;
  }}

  .evo-baseline {{
    background: #fafafa;
    border-right: 1px dashed #cbd5e1;
  }}

  .evo-production {{
    background: #f8fafc;
  }}

  .evo-col-title {{
    font-size: 8.8pt;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 8px;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 4px;
  }}

  .evo-col ul {{
    margin: 0 0 10px 14px;
    padding: 0;
    font-size: 8.2pt;
    color: #334155;
  }}

  .evo-col li {{
    margin-bottom: 3px;
  }}

  .evo-score-box {{
    padding: 6px 10px;
    border-radius: 5px;
    display: flex;
    justify-content: space-between;
    align-items: baseline;
  }}

  .evo-score-base {{
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
  }}

  .evo-score-prod {{
    background: #eff6ff;
    border: 1.5px solid #93c5fd;
  }}

  .evo-score-num {{
    font-family: 'JetBrains Mono', monospace;
    font-weight: 800;
    font-size: 13pt;
    color: #0f172a;
  }}

  .evo-score-prod .evo-score-num {{
    color: #1d4ed8;
  }}

  .evo-score-label {{
    font-size: 7.5pt;
    font-weight: 600;
    color: #475569;
  }}

  /* Flowchart */
  .flowchart-container {{
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-radius: 7px;
    padding: 12px 14px;
    margin: 14px 0;
    text-align: center;
    page-break-inside: avoid;
    break-inside: avoid;
  }}

  .fc-step {{
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 5px;
    padding: 6px 10px;
    display: inline-block;
    font-size: 8.2pt;
    font-weight: 600;
    color: #0f172a;
    margin: 2px 0;
  }}

  .fc-input {{
    background: #eff6ff;
    border-color: #93c5fd;
    color: #1d4ed8;
  }}

  .fc-output {{
    background: #f0fdf4;
    border-color: #86efac;
    color: #166534;
  }}

  .fc-arrow {{
    font-size: 10pt;
    color: #64748b;
    margin: 1px 0;
    line-height: 1;
  }}

  .fc-split {{
    display: grid;
    grid-template-columns: 1.3fr 1fr;
    gap: 8px;
    margin: 6px 0;
    text-align: left;
  }}

  .fc-branch {{
    border-radius: 5px;
    padding: 8px 10px;
    font-size: 7.8pt;
  }}

  .fc-high {{
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
  }}

  .fc-low {{
    background: #fffbeb;
    border: 1px solid #fde68a;
  }}

  .fc-branch-title {{
    font-weight: 700;
    margin-bottom: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 7.8pt;
  }}

  .fc-high .fc-branch-title {{ color: #166534; }}
  .fc-low .fc-branch-title {{ color: #92400e; }}

  .fc-substep {{
    margin-bottom: 3px;
    color: #334155;
  }}

  .fc-tier {{
    margin-left: 8px;
    color: #475569;
    font-size: 7.5pt;
  }}

  /* Question Cards */
  .question-card {{
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    margin: 12px 0;
    padding: 10px 12px;
    page-break-inside: avoid;
    break-inside: avoid;
  }}

  .q-header {{
    display: flex;
    align-items: baseline;
    gap: 7px;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 5px;
    margin-bottom: 7px;
  }}

  .q-header h4 {{
    margin: 0;
    font-size: 9.5pt;
    font-weight: 800;
    color: #0f172a;
  }}

  .q-badge {{
    font-size: 6.8pt;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    text-transform: uppercase;
    color: #2563eb;
    background: #dbeafe;
    padding: 2px 5px;
    border-radius: 3px;
    flex-shrink: 0;
  }}

  .qa-row {{
    margin: 5px 0;
    padding: 5px 7px;
    border-radius: 4px;
    font-size: 8.5pt;
  }}

  .qa-row p {{
    margin: 2px 0 0 0;
    color: #1e293b;
    line-height: 1.42;
  }}

  .qa-tag {{
    font-size: 6.5pt;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 1px 4px;
    border-radius: 2px;
    display: inline-block;
  }}

  .qa-short {{ background: #eff6ff; border-left: 3px solid #3b82f6; }}
  .tag-short {{ background: #bfdbfe; color: #1d4ed8; }}

  .qa-deep {{ background: #f8fafc; border-left: 3px solid #64748b; }}
  .tag-deep {{ background: #e2e8f0; color: #334155; }}

  .qa-followup-q {{ background: #fefce8; border-left: 3px solid #eab308; }}
  .tag-followup {{ background: #fef08a; color: #854d0e; }}

  .qa-followup-a {{ background: #f8fafc; border-left: 3px solid #94a3b8; margin-left: 10px; }}
  .tag-fanswer {{ background: #e2e8f0; color: #475569; }}

  .qa-guardrail {{ background: #fff1f2; border-left: 3px solid #e11d48; }}
  .tag-warning {{ background: #fecdd3; color: #9f1239; }}

  /* ADR Cards */
  .adr-card {{
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-left: 3.5px solid #2563eb;
    border-radius: 5px;
    padding: 9px 11px;
    margin: 10px 0;
    page-break-inside: avoid;
    break-inside: avoid;
  }}

  .adr-header {{
    display: flex;
    align-items: baseline;
    gap: 7px;
    margin-bottom: 6px;
    border-bottom: 1px dashed #cbd5e1;
    padding-bottom: 4px;
  }}

  .adr-header h4 {{
    margin: 0;
    font-size: 9.3pt;
    font-weight: 700;
  }}

  .adr-badge {{
    font-size: 6.8pt;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    text-transform: uppercase;
    background: #dbeafe;
    color: #1e40af;
    padding: 1px 5px;
    border-radius: 3px;
  }}

  .adr-item {{
    font-size: 8.4pt;
    margin-bottom: 3px;
    line-height: 1.4;
  }}

  .adr-label {{
    font-weight: 700;
    color: #0f172a;
    display: inline-block;
    min-width: 140px;
  }}

  .adr-val {{
    color: #1e293b;
  }}

  /* Incident Cards */
  .incident-card {{
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 5px;
    padding: 10px 12px;
    margin: 12px 0;
    page-break-inside: avoid;
    break-inside: avoid;
  }}

  .inc-header {{
    display: flex;
    align-items: baseline;
    gap: 7px;
    border-bottom: 1.5px solid #0f172a;
    padding-bottom: 4px;
    margin-bottom: 7px;
  }}

  .inc-header h4 {{
    margin: 0;
    font-size: 9.5pt;
    font-weight: 800;
    color: #0f172a;
  }}

  .inc-badge {{
    font-size: 6.8pt;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    text-transform: uppercase;
    background: #fed7aa;
    color: #9a3412;
    padding: 1px 5px;
    border-radius: 3px;
  }}

  .inc-step {{
    margin: 5px 0;
    padding: 4px 7px;
    border-radius: 4px;
    font-size: 8.4pt;
  }}

  .inc-step p {{
    margin: 2px 0 0 0;
    line-height: 1.4;
  }}

  .inc-step-tag {{
    font-size: 6.5pt;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    text-transform: uppercase;
    padding: 1px 4px;
    border-radius: 2px;
    display: inline-block;
  }}

  .step-problem {{ background: #fff1f2; border-left: 3px solid #e11d48; }}
  .tag-prob {{ background: #fecdd3; color: #9f1239; }}

  .step-cause {{ background: #fefce8; border-left: 3px solid #ca8a04; }}
  .tag-cause {{ background: #fef08a; color: #854d0e; }}

  .step-fix {{ background: #eff6ff; border-left: 3px solid #2563eb; }}
  .tag-fix {{ background: #dbeafe; color: #1e40af; }}

  .step-res {{ background: #f0fdf4; border-left: 3px solid #16a34a; }}
  .tag-res {{ background: #dcfce7; color: #166534; }}

  .step-lesson {{ background: #faf5ff; border-left: 3px solid #9333ea; }}
  .tag-lesson {{ background: #f3e8ff; color: #6b21a8; }}

  .inc-substep {{
    font-size: 8.2pt;
    margin: 2px 0 2px 12px;
    color: #334155;
  }}

</style>
</head>
<body>
{body_html}
</body>
</html>
"""

def main():
    if not MD_PATH.exists():
        raise FileNotFoundError(f"Markdown file not found: {MD_PATH}")
    
    print(f"Reading Markdown from: {MD_PATH}")
    md_text = MD_PATH.read_text(encoding="utf-8")
    
    print("Parsing Markdown into structured HTML...")
    body_html = parse_markdown_to_html(md_text)
    full_html = build_html_document(body_html)
    
    HTML_PATH.write_text(full_html, encoding="utf-8")
    print(f"Written HTML to: {HTML_PATH} ({len(full_html)} chars)")
    
    browser_paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    browser_exe = next((bp for bp in browser_paths if os.path.exists(bp)), None)
    if not browser_exe:
        raise RuntimeError("No headless browser found for PDF generation.")
        
    print(f"Using browser: {browser_exe}")
    cmd = [
        browser_exe,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={PDF_PATH}",
        str(HTML_PATH.resolve()),
    ]
    
    print("Compiling PDF with headless browser...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Browser error: {res.stderr}")
        raise RuntimeError("PDF generation command failed.")
        
    if PDF_PATH.exists() and PDF_PATH.stat().st_size > 0:
        size = PDF_PATH.stat().st_size
        print(f"SUCCESS: Generated PDF at {PDF_PATH} ({size:,} bytes)")
        shutil.copyfile(PDF_PATH, DOCS_ALT_PDF_PATH)
        shutil.copyfile(PDF_PATH, ROOT_PDF_PATH)
        print(f"Copied PDF to: {DOCS_ALT_PDF_PATH}")
        print(f"Copied PDF to: {ROOT_PDF_PATH}")
    else:
        raise RuntimeError("PDF was not created or is empty.")

if __name__ == "__main__":
    main()

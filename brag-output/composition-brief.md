# Hyperframes Composition Brief: WeMentors Academy AI Assistant

## Objective
Create a short, polished, high-energy launch-style brag video for WeMentors Academy AI Assistant.

## Output
- Composition directory: `brag-output/composition/`
- Rendered video: `brag-output/brag.mp4`
- Format: landscape — 1920x1080
- Duration: 20 seconds

## Source Material
- Project root: `c:\Users\Raaid\Downloads\wementorsexperiment\wementorschatbotexperiment`
- Primary files read: `index.html`, `README.md`, `wm_brand_logo.png`, `PROJECT_NOTES.md`
- Product name: WeMentors Academy
- Tagline / strongest claim: "Personalized Learning, Proven Results. 1:1 Live Online Academic Mentoring & Confident Speaker."
- Key UI or visual moment to recreate: The authentic dark glassmorphic `#chatbotPanel` with glowing gradient accents, green pulse status dot, user/assistant chat bubbles, and suggestion pills.
- Copy that must appear verbatim:
  - "Personalized Learning, Proven Results."
  - "Hi! I'm the WeMentors Assistant. I can help you explore our programs, understand how mentoring works, and guide you toward booking a free demo."
  - "Tell me about Confident Speaker"
  - "Dual-Model Failover: Gemini 2.5 Flash-Lite + Groq Llama 3.3 70B"
  - "https://wementors.vercel.app"

## Creative Direction
- Tone preset: `polished`
- Creative direction: "Sleek, high-energy modern AI product launch"
- Interpretation: Fast-moving but highly legible; every critical claim holds for >= 1.2s; sharp glassmorphism and subtle lighting glows.
- Angle: Showcase a production-hardened AI assistant that powers academic mentoring with deterministic safeguards and dual-model redundancy.
- Hook: The golden brand promise "Personalized Learning. Proven Results." bursting into life with clean typography.
- Outro / punchline: "Production-Ready. Defensible. Live at wementors.vercel.app."
- Avoid:
  - Generic SaaS language
  - Abstract filler visuals
  - Unrelated visual redesign

## Visual Identity
- Background: `#0f0e1a` / `#141225` (Dark Glass 900)
- Text: `#f3f1fb` (Hi contrast) / `#a9a3c4` (Dim)
- Accent: `linear-gradient(135deg, #8b6bff, #5b7fff)` (Glass Accent Gradient) & `#f6d43f` / `#c9971c` (Gold)
- Display font: `Fraunces`, Georgia, serif
- Body font: `Inter`, system-ui, -apple-system, sans-serif
- Visual references from the project: `#chatbotPanel`, `.chatbot-message`, `.chatbot-user`, `.chatbot-assistant`, `.chatbot-suggestion`, `wm_brand_logo.png`

## Storyboard
Use the storyboard in `brag-output/brag-plan.md` as the creative contract.

Scene summary:
1. Brand Hook — 3.5s — Golden brand headline & logo reveal.
2. Meet The AI Assistant — 5.0s — Dark glassmorphic chat widget slides up, welcome message & suggestion pills.
3. Live Intelligence & Dual-Model Failover — 6.0s — User query, typing animation, assistant response, and architecture badges.
4. Production Ready & Live CTA — 5.5s — 20/20 test suite validation, high-speed edge deployment, and live URL card.

## Audio
- Audio role: Warm, driving electronic groove with clean UI feedback.
- Audio arc: Building intro, sustained rhythmic drive through the conversation, triumphant resolution on the live link.
- Music: `assets/music/bgm.mp3`
- Music treatment: Volume at 0.65, soft fade-in over 0.5s, clean fade-out over 1.5s at end.
- Music cue guidance: 120.19 BPM (~0.5s beat interval); scene transitions lock cleanly to beat grid at 3.5s, 8.5s, 14.5s.
- Audio-reactive treatment: Subtle rhythmic glow on the chat panel border and status badge.
- Audio-coupled moments:
  - 3.6s: Panel slide-in pop (`assets/sfx/pop.ogg`)
  - 8.7s: User message send click (`assets/sfx/send.ogg`)
  - 10.1s: Assistant response chime (`assets/sfx/chime.ogg`)
  - 14.8s: Outro badge lock pop (`assets/sfx/click.ogg`)
- SFX files: Copied into `brag-output/composition/assets/sfx/`.

## Hyperframes Instructions
- Composition directory: `brag-output/composition/`
- Render target: `brag-output/brag.mp4`
- Resolution: 1920x1080 (16:9)
- Run `npx hyperframes check` before rendering.

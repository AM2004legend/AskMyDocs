# Paste these into Claude Code, one at a time

## Before you start
1. Create an empty `askmydocs` folder, open it in VS Code.
2. Put CLAUDE.md in the root, PRD.md in `docs/`, and your working `rag.py` in the root.
3. Create `.env` yourself (never paste your key into Claude Code's chat):
   ```
   LLM_BASE_URL=https://api.groq.com/openai/v1
   LLM_API_KEY=your_key_here
   LLM_MODEL=pick_a_current_model_from_console.groq.com
   ```
4. Run `git init`, and make sure `.env` is in `.gitignore` before your first commit.

## Prompt 1 — plan only

Read CLAUDE.md, docs/PRD.md and rag.py fully. Don't write code yet.

Give me:
1. A one-paragraph summary of what we're building, in your own words.
2. What from rag.py you'll reuse and where it will live in the new structure.
3. The Phase 1 plan: every file and package, and why, in a short table.
4. Anything unclear or risky in the PRD (especially Groq rate limits and long books).
5. What I need installed on Windows (Python 3.12, Docker Desktop) and PowerShell commands to check.

Wait for my "go".

## Prompt 2 — Phase 1

Go ahead with Phase 1 only. Confirm `.env` is gitignored and `.env.example` has no real key.
Run ruff and pytest, show the output, show me how to start Postgres and hit /health, then explain
what you built, the key concepts and 2 interview questions. Stop.

## Prompt for every later phase

Do Phase N from CLAUDE.md. Plan first (files + why), wait for my go, build with tests, run them,
explain the same way. Stop.

## Prompt for Phase 5 (evaluation)

Help me build eval/questions.yaml. Download one public-domain novel from Project Gutenberg into
samples/. Draft 10 questions for it following the PRD mix, with expected answers and the chapter
where each answer is. I'll check and correct them before we run anything.

## Prompt for Phase 9 (PII redaction)

Do Phase 9 from the PRD. Start with the detectors and the tripwire, with tests using synthetic PII
only (fake names, fake PAN/Aadhaar-format numbers that pass checksums). Show me a sample document
before and after redaction, and the exact payload that would be sent to Groq. Then the vault,
restore step, and UI toggle.

## Handy prompts

- "Explain this file line by line like I'll be asked about it in an interview."
- "Run the eval and show me a before/after table for what we just changed."
- "Try to break the prompt-injection guard with a malicious document. What happens?"
- "Quiz me with 5 questions on this phase. Don't answer until I reply."
- "This failed: <paste error>. Explain the cause before fixing it."
- "Review this phase like a senior ML engineer. What would they criticise?"

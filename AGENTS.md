# Project Instructions for AI Agents

## Committing
- Do NOT commit, push, or create PRs unless explicitly prompted by the user.
- When committing is requested, inspect `git status`, `git diff`, and `git log --oneline -10` first.
- Stage only intended files. Never commit secrets, generated artifacts, or test output.
- When dispatching subagents, ensure instructions do NOT include commit steps without user approval.

## Running code
- When running Python code that does heavy computation (multiprocessing, large arrays), cap worker count at 10 to leave room for the user's desktop. Use `min(10, os.cpu_count() or 10)` for Pool size.
- Always keep terminal output clean — no runaway progress bars or verbose debug logs.

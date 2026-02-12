# prompt-to-drawio

A CLI-first Codex skill to generate/edit/export/validate draw.io diagrams from natural language prompts, without any frontend.

## What's Inside

- `SKILL.md`: skill entry instructions
- `scripts/prompt_to_drawio.py`: runtime CLI
- `references/`: capability and rendering notes
- `agents/openai.yaml`: assistant interface metadata

## Core Capabilities

- Prompt -> `.drawio`
- Prompt-driven edit for existing `.drawio`
- Export to `png/svg/pdf/jpg`
- Context ingestion from file/url/image/pdf
- Shape-library lookup
- Visual validation loop

## CLI Commands

```bash
python3 scripts/prompt_to_drawio.py generate ...
python3 scripts/prompt_to_drawio.py edit ...
python3 scripts/prompt_to_drawio.py export ...
python3 scripts/prompt_to_drawio.py validate ...
python3 scripts/prompt_to_drawio.py library --list
```

## Recent Hardening

- Effective config summary at startup
- `.env` controls: `--no-dotenv`, `--dotenv-file`
- Model preflight with fallback and warnings
- JSON recovery for validation/edit outputs
- `--validate-soft` to avoid failing pipeline after files already generated

## Requirements

- Python 3.9+
- draw.io CLI (`drawio`) for image export (or Docker fallback)
- OpenAI-compatible API endpoint for generation/validation

## License

MIT

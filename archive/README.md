# Archive - Old Files

This folder contains the original files from before the codebase reorganization.

## Files

- **`agent_basic.py`** - Original monolithic script containing agent setup, GCG algorithm, and execution logic
- **`tools.py`** - Original tool definitions (moved to `src/agent/tools.py`)
- **`scoring.py`** - Original scoring functions (moved to `src/scoring/leak_score.py`)

## Migration

These files have been replaced by the new modular structure in the `src/` directory:

- `agent_basic.py` → Split into:
  - `src/agent/setup.py` - Agent creation
  - `src/attacks/gcg.py` - GCG algorithm
  - `src/config/` - Configuration
  - `main.py` - Entry point

- `tools.py` → `src/agent/tools.py`
- `scoring.py` → `src/scoring/leak_score.py`

## Purpose

These files are kept for:
- Reference and comparison
- Understanding the migration
- Historical record

**Note:** These files are no longer used. Use the new structure in `src/` instead.

See [../docs/migration-guide.md](../docs/migration-guide.md) for details on the migration.


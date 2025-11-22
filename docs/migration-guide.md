# Codebase Reorganization - Migration Guide

## Overview

The codebase has been reorganized from a flat structure to a professional package structure for better maintainability, testability, and extensibility.

## What Changed

### Old Structure
```
agentTest/
├── agent_basic.py      # Everything in one file
├── tools.py
├── scoring.py
└── requirements.txt
```

### New Structure
```
agentTest/
├── src/                # Main package
│   ├── agent/         # Agent setup and tools
│   ├── attacks/       # GCG attack implementation
│   ├── scoring/       # Leakage scoring
│   ├── config/        # Configuration
│   └── utils/         # Utilities
├── main.py            # Clean entry point
└── tests/             # Test directory
```

## File Mapping

| Old File | New Location | Notes |
|----------|--------------|-------|
| `agent_basic.py` | Split into multiple files | See breakdown below |
| `tools.py` | `src/agent/tools.py` | Moved to agent package |
| `scoring.py` | `src/scoring/leak_score.py` | Moved to scoring package |

### Breakdown of `agent_basic.py`

The monolithic `agent_basic.py` has been split into:

1. **Agent Setup** → `src/agent/setup.py`
   - `create_agent()` function
   - `run_agent_with_suffix()` function
   - LLM initialization

2. **Middleware** → `src/agent/middleware.py`
   - `handle_tool_errors()` function

3. **GCG Algorithm** → `src/attacks/gcg.py`
   - `GCGAttack` class
   - `gcg_step()` method
   - `run()` method

4. **Vocabulary** → `src/attacks/vocab.py`
   - `VOCAB` constant

5. **Configuration** → `src/config/`
   - `settings.py`: All configurable settings
   - `prompts.py`: System prompts and base queries

6. **Entry Point** → `main.py`
   - Clean, simple main function
   - Easy to understand flow

## How to Use the New Structure

### Running the Attack

**Old way:**
```bash
python agent_basic.py
```

**New way:**
```bash
python main.py
```

### Importing Modules

**Old way:**
```python
from tools import add, subtract
from scoring import leak_score
```

**New way:**
```python
from src.agent import add, subtract
from src.scoring import leak_score
```

Or if you're inside the package:
```python
from .agent import add, subtract
from .scoring import leak_score
```

### Configuration

**Old way:**
```python
# Hardcoded in agent_basic.py
max_iters = 30
suffix_len = 15
```

**New way:**
```python
from src.config import Settings

settings = Settings(
    max_iters=30,
    suffix_len=15
)
```

### Using GCG Attack

**Old way:**
```python
run_gcg(max_iters=30, suffix_len=15)
```

**New way:**
```python
from src.attacks import GCGAttack
from src.config import Settings

settings = Settings(max_iters=30, suffix_len=15)
gcg = GCGAttack(score_function=score_suffix, settings=settings)
best_suffix, best_score = gcg.run()
```

## Benefits of New Structure

1. **Modularity**: Each component has a clear purpose
2. **Testability**: Easy to test individual components
3. **Maintainability**: Changes are localized to specific modules
4. **Extensibility**: Easy to add new attacks or agents
5. **Configuration**: Centralized, easy to modify
6. **Professional**: Follows Python package best practices

## Migration Steps

1. ✅ New structure created
2. ✅ Code refactored and split
3. ✅ Imports updated
4. ✅ Main entry point created
5. ✅ README updated
6. ⚠️ **Old files kept for reference** - You can delete them when ready:
   - `agent_basic.py` (replaced by `main.py` + modules)
   - `tools.py` (moved to `src/agent/tools.py`)
   - `scoring.py` (moved to `src/scoring/leak_score.py`)

## Testing the New Structure

To verify everything works:

```bash
# Test imports
python -c "from src.agent import create_agent; print('Agent import OK')"
python -c "from src.attacks import GCGAttack; print('GCG import OK')"
python -c "from src.scoring import leak_score; print('Scoring import OK')"

# Run the main script
python main.py
```

## Next Steps

1. **Delete old files** (optional, after verifying new structure works):
   ```bash
   rm agent_basic.py tools.py scoring.py
   ```

2. **Add tests** (recommended):
   - Create test files in `tests/` directory
   - Test individual components
   - Test integration

3. **Add CLI arguments** (optional):
   - Use `argparse` in `main.py`
   - Allow command-line configuration

4. **Add logging** (optional):
   - Replace print statements with proper logging
   - Configurable log levels

## Questions?

If you encounter any issues with the new structure, check:
- Import paths are correct
- All dependencies are installed
- Python path includes the project root


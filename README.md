# Book Translator v2

A sophisticated book translation system that uses LLM to translate books while maintaining consistency and preserving document structure.

## Features

- **Hierarchical Structure Detection**: Automatically detects chapters, sections, and subsections
- **Smart Chunking**: Intelligently splits text into manageable chunks without breaking paragraphs
- **Translation Memory**: Maintains consistency for character names, locations, terminology, and idioms
- **Genre-Aware Translation**: Customizable translation rules based on text genre
- **Language-Specific Instructions**: Tailored translation guidelines for target languages
- **Resumable Translation**: Checkpoint system for resuming interrupted translations
- **Multiple Output Formats**: Supports JSON and Markdown output

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

1. Create a .env file:

2. Edit `.env` and add your LLM Gateway credentials:
```
LLM_GATEWAY_CLIENT_ID=your_client_id_here
LLM_GATEWAY_CLIENT_SECRET=your_client_secret_here
```

## Usage

### Basic Translation

```bash
python translator.py input.txt -o output.json
```

### With Options

```bash
python translator.py input.txt \
  -o translation.md \
  -f markdown \
  -g philosophy \
  --source-lang English \
  --target-lang Chinese \
  --chunk-size 8192 \
  -v
```

### Resume from Checkpoint

```bash
python translator.py input.txt --resume translation_5
```

## Command Line Arguments

- `source`: Source text file (.txt)
- `-o, --output`: Output file path
- `-f, --format`: Output format (json, markdown, md)
- `-g, --genre`: Genre of the text (default: philosophy)
- `--source-lang`: Source language (default: English)
- `--target-lang`: Target language (default: Chinese)
- `--chunk-size`: Chunk size in tokens (default: 8192)
- `--no-cache`: Disable caching
- `--resume`: Resume from checkpoint name
- `--clear-memory`: Clear translation memory before starting
- `-v, --verbose`: Enable verbose logging

## Project Structure

```
book_translator_v2/
├── translator.py           # Main entry point and orchestrator
├── text_processor.py       # Text chunking and hierarchy detection
├── llm_gateway.py         # LLM interface
├── file_handler.py        # File I/O and formatting
├── config.py              # Configuration settings
├── tools/
│   └── memo.py            # Translation memory management
├── prompts/
│   ├── hierarchy.txt      # Hierarchy detection prompt
│   ├── translate.txt      # Translation prompt
│   ├── genre/
│   │   └── philosophy.txt # Philosophy genre guidelines
│   └── lang/
│       └── chinese.txt    # Chinese language instructions
└── .cache/               # Cache and checkpoint files

```

## Output Files

The system generates two main output files:

1. **Translation File** (`output.json` or `output.md`): Contains the translated text with preserved hierarchy
2. **Glossary File** (`output.glossary.md`): Contains all tracked terms and their translations

## Translation Memory

The system maintains a translation memory that tracks:
- Character names and their consistent translations
- Location names
- Technical terminology
- Idiomatic expressions
- Context-aware translations

## Caching and Checkpoints

- Automatic caching of translation progress
- Checkpoint system for resuming interrupted translations
- Cache files stored in `.cache/` directory

## Customization

### Adding New Genres

Create a new file in `prompts/genre/` with genre-specific translation guidelines.

### Adding New Languages

Create a new file in `prompts/lang/` with language-specific instructions.

## License

MIT License - See LICENSE file for details
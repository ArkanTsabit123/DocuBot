# DocuBot - Personal AI Knowledge Assistant

## What is DocuBot?
DocuBot is a 100% local, privacy-focused AI assistant that transforms your personal document collection into a searchable knowledge base. It runs completely offline on your own computer, keeping all your data private while providing AI-powered document querying capabilities.

## Key Features
- **Complete Privacy**: All processing happens locally on your device - no data ever leaves your computer
- **Free & Open Source**: No subscriptions, no fees, completely free to use forever
- **Offline-First Design**: Works without any internet connection after initial setup
- **Multi-Format Support**: Handles PDFs, Word documents, ebooks (EPUB), text files, web articles, images with OCR, and more
- **Natural Language Queries**: Ask questions about your documents in plain English
- **Citation & Sources**: Every answer includes references to the original documents
- **Conversational Interface**: Chat with your documents as if talking to a research assistant

## Quick Start Guide

### System Requirements
- **Minimum**: Windows 10/11, macOS 10.15+, Ubuntu 20.04+ | 8GB RAM | 10GB free space
- **Recommended**: 16GB RAM | SSD storage | Optional GPU for faster processing

### Installation Methods

#### Method 1: One-Click Installer (Recommended)
Download the appropriate installer for your system:
- **Windows**: `DocuBot-Setup.exe` (Includes all dependencies)
- **macOS**: `DocuBot.dmg` (Drag to Applications folder)
- **Linux**: `docubot.AppImage` (Make executable and run)

#### Method 2: Python Package
```bash
# Install via pip
pip install docubot

# Run the application
docubot
```

#### Method 3: From Source
```bash
# Clone the repository
git clone https://github.com/yourusername/docubot.git
cd docubot

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch DocuBot
python app.py
```

### First-Time Setup
1. **Initial Launch**: DocuBot will guide you through downloading required AI models (3-5GB download)
2. **Model Selection**: Choose between:
   - **Llama2-7B**: Balanced performance (default)
   - **Mistral-7B**: Faster responses
   - **Neural-Chat-7B**: More accurate answers
3. **Document Folder**: Select where your documents are stored or drag & drop files directly

## How to Use DocuBot

### Adding Documents
1. **Drag & Drop**: Simply drag files into the DocuBot window
2. **Folder Monitoring**: Set up auto-import from specific folders
3. **Batch Import**: Select multiple files at once
4. **Web Content**: Paste URLs to save web articles

### Asking Questions
Type natural language questions about your documents:
- "What are the main arguments in this research paper?"
- "Summarize the key points from my meeting notes"
- "Find all references to machine learning in these documents"
- "Compare the approaches in document A and document B"

### Document Management
- **Collections**: Organize documents into custom collections
- **Tags**: Automatic and manual tagging system
- **Search**: Full-text search across all documents
- **Export**: Save conversations as PDF, Markdown, or text files

## Supported File Formats
| Format | Features |
|--------|----------|
| **PDF** | Text extraction, embedded images, annotations |
| **DOCX** | Full formatting preservation, tables, headers |
| **EPUB** | Ebook support, chapter navigation |
| **TXT/Markdown** | Clean text processing |
| **HTML** | Web content saving |
| **Images (JPG/PNG)** | OCR text extraction (English, Indonesian) |
| **CSV/Excel** | Tabular data processing |
| **PowerPoint** | Slide content extraction |

## Configuration Options

### AI Model Settings
Edit `~/.docubot/config/app_config.yaml`:

```yaml
ai:
  llm:
    model: "llama2:7b"  # Options: llama2:7b, mistral:7b, neural-chat:7b
    temperature: 0.1     # Lower = more focused, Higher = more creative
    max_tokens: 1024     # Response length limit
  
  embeddings:
    model: "all-MiniLM-L6-v2"  # Options: all-mpnet-base-v2 (more accurate)
  
  rag:
    top_k: 5                    # Number of document chunks to consider
    similarity_threshold: 0.7   # Relevance cutoff
```

### Performance Tuning
```yaml
performance:
  max_workers: 4           # Parallel processing threads
  cache_enabled: true      # Enable response caching
  cache_size_mb: 500       # Cache size limit
  enable_monitoring: true  # Resource usage tracking
```

## Advanced Features

### Command Line Interface
```bash
# Process documents via CLI
docubot process /path/to/documents/

# Ask questions from terminal
docubot ask "What is this document about?" --file document.pdf

# Export your knowledge base
docubot export --format json --output knowledge_base.json

# Monitor system resources
docubot status
```

### Plugin System
Extend DocuBot with plugins:
- **Obsidian Sync**: Sync with Obsidian vaults
- **Notion Export**: Export to Notion databases
- **Browser Clipper**: Save web content directly from browser
- **Voice Interface**: Voice commands and responses

### API Access (Developer Feature)
Run DocuBot as a local API server:
```bash
docubot serve --port 8000
```

Then query via HTTP:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is in my documents?", "documents": ["doc1.pdf"]}'
```

## Troubleshooting

### Common Issues

#### Slow Performance
- Reduce `top_k` value in config (try 3 instead of 5)
- Switch to faster embedding model (`all-MiniLM-L6-v2`)
- Enable GPU acceleration if available
- Increase system RAM allocation

#### Memory Issues
- Process documents in smaller batches
- Reduce `chunk_size` from 500 to 300 tokens
- Clear cache: `docubot cache --clear`

#### Model Download Problems
- Manual download: Visit [ollama.ai/library](https://ollama.ai/library)
- Alternative models: Use `--model mistral:7b` (smaller download)

### Getting Help
1. Check the [Troubleshooting Guide](docs/troubleshooting/)
2. Search [GitHub Issues](https://github.com/yourusername/docubot/issues)
3. Join [Community Discussions](https://github.com/yourusername/docubot/discussions)

## Development & Contribution

### Building from Source
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Build executable
python scripts/build_windows.py  # or build_mac.py / build_linux.py
```

### Project Structure
```
DocuBot/
├── src/                 # Source code
│   ├── core/           # Application logic
│   ├── ai_engine/      # AI/ML components
│   ├── document_processing/  # File handlers
│   ├── ui/             # User interfaces
│   └── plugins/        # Extensions
├── tests/              # Test suite
├── docs/               # Documentation
└── scripts/            # Build & utility scripts
```

### Contributing Guidelines
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure code passes all tests
5. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## Privacy & Security
- **No Telemetry**: DocuBot does not collect any usage data
- **Local Processing**: All AI processing happens on your device
- **Encryption Option**: Optional AES-256 encryption for sensitive documents
- **Open Source**: Full code transparency

## Roadmap
### Next Release (v1.1)
- Mobile companion app
- Browser extension
- Advanced search filters
- Multi-language interface

### Future Plans
- Voice interface
- Collaborative features
- Custom model training
- Cloud sync (optional)

## License
DocuBot is released under the MIT License. See [LICENSE](LICENSE) file for details.

## Support & Community
- **Documentation**: [docs.docubot.ai](https://docs.docubot.ai)
- **GitHub**: [github.com/ArkanTsabit123/DocuBot](https://github.com/yourusername/docubot)
- **Discord**: [Join Community](https://discord.gg/docubot)
- **Email**: arkantsabit@gmail.com

---

**DocuBot**: Your documents, your AI, your privacy. Always local, always free.
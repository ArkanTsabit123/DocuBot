# Contributing to DocuBot

Thank you for your interest in contributing to DocuBot! This document provides guidelines and instructions for contributing to this open-source project. Whether you're fixing bugs, implementing new features, improving documentation, or helping with translations, your contributions are welcome.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Project Structure](#project-structure)
- [Coding Standards](#coding-standards)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)
- [Feature Requests](#feature-requests)
- [Documentation](#documentation)
- [Testing](#testing)
- [Release Process](#release-process)
- [Community](#community)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. By participating in this project, you agree to:

- Be respectful and constructive in all communications
- Focus on what is best for the community
- Accept constructive criticism gracefully
- Show empathy towards other community members
- Use welcoming and inclusive language

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by contacting the project team at arkantsabit025@gmail.com

## Getting Started

### Prerequisites
- Python 3.11 or higher
- Git
- Basic understanding of Python development
- Familiarity with AI/ML concepts (helpful but not required)

### First-Time Contributors
1. **Fork the Repository**: Click the "Fork" button on GitHub
2. **Clone Your Fork**:
   ```bash
   git clone https://github.com/ArkanTsabit123/DocuBot.git
   cd docubot
   ```
3. **Set Upstream Remote**:
   ```bash
   git remote add upstream https://github.com/ArkanTsabit123/DocuBot.git
   ```
4. **Create a Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

### Good First Issues
New contributors should check issues labeled with:
- `good first issue`
- `beginner-friendly`
- `help wanted`

## Development Environment

### Setting Up Development Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install development dependencies
pip install -r requirements-dev.txt
pip install -e .

# Install pre-commit hooks
pre-commit install
```

### Required Tools
- **Code Editor**: VS Code (recommended) or PyCharm
- **Git**: Latest version
- **Python Tools**:
  - `black` for code formatting
  - `flake8` for linting
  - `mypy` for type checking
  - `pytest` for testing

### Configuration Files
Create a local development configuration file:
```bash
cp config/app_config.example.yaml config/app_config.local.yaml
```

Edit `app_config.local.yaml` for development settings:
```yaml
development:
  debug: true
  log_level: DEBUG
  test_mode: true
```

## Project Structure

### Key Directories
```
src/
├── core/              # Core application logic
├── document_processing/  # Document handling
├── ai_engine/         # AI/ML components
├── vector_store/      # Vector database layer
├── database/          # Metadata storage
├── ui/               # User interfaces
├── storage/          # Data management
├── utilities/        # Helper functions
└── plugins/          # Plugin system
```

### Architecture Overview
- **Document Flow**: Document → Processor → Chunker → Embedder → Vector Store
- **Query Flow**: Query → Embedder → Vector Search → LLM → Response
- **Data Storage**: SQLite (metadata) + ChromaDB (vectors)

## Coding Standards

### Python Style Guide
We follow [PEP 8](https://pep8.org/) with specific additions:

#### Formatting
```python
# Use type hints for all functions
def process_document(file_path: Path) -> Dict[str, Any]:
    """Brief description of function.
    
    Args:
        file_path: Path to document file
        
    Returns:
        Dictionary with processing results
        
    Raises:
        ValueError: If format unsupported
        IOError: If file cannot be read
    """
    pass

# Use meaningful variable names
document_chunks = []  # Good
dc = []               # Avoid

# Limit line length to 88 characters (Black default)
```

#### Import Order
```python
# 1. Standard library imports
import os
import sys
from typing import Dict, List, Optional

# 2. Third-party imports
import numpy as np
from sentence_transformers import SentenceTransformer

# 3. Local application imports
from .config import Config
from .processor import DocumentProcessor
```

#### Error Handling
```python
try:
    result = process_document(file_path)
except FileNotFoundError as e:
    logger.error(f"File not found: {file_path}")
    raise
except PermissionError as e:
    logger.error(f"Permission denied: {file_path}")
    raise
except Exception as e:
    logger.error(f"Unexpected error processing {file_path}: {e}")
    raise CustomError("Document processing failed") from e
```

### Documentation Standards
- All public functions/methods must have docstrings
- Use Google-style docstring format
- Include examples for complex functions
- Update README.md when adding new features

### Commit Messages
Use [Conventional Commits](https://www.conventionalcommits.org/):
```
feat: add EPUB document support
fix: resolve memory leak in document processor
docs: update installation instructions
test: add unit tests for PDF extractor
refactor: simplify configuration loading
chore: update dependencies
```

Example:
```bash
git commit -m "feat: add support for EPUB document format"
```

## Pull Request Process

### Before Submitting a PR
1. **Sync with Upstream**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```
2. **Run Tests**:
   ```bash
   pytest
   pre-commit run --all-files
   ```
3. **Check Coverage**:
   ```bash
   pytest --cov=src --cov-report=html
   ```

### PR Checklist
- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] No new warnings/errors
- [ ] Branch is up to date with main
- [ ] Commit messages follow conventions

### PR Template
When creating a PR, please include:
1. **Description**: What does this PR do?
2. **Related Issue**: Link to issue (if applicable)
3. **Type of Change**:
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Breaking change
   - [ ] Documentation update
4. **Testing**:
   - [ ] Unit tests added
   - [ ] Integration tests added
   - [ ] Manual testing performed
5. **Screenshots** (if UI changes)

### Review Process
1. **Initial Review**: Within 2-3 business days
2. **Feedback**: Address all review comments
3. **Approval**: Requires at least one maintainer approval
4. **Merge**: Squash and merge (unless otherwise specified)

## Issue Guidelines

### Creating Issues
Use the issue templates provided:
- **Bug Report**: For reporting bugs
- **Feature Request**: For suggesting new features
- **Documentation**: For documentation improvements

### Bug Reports
Include:
1. **Description**: Clear description of the bug
2. **Steps to Reproduce**:
   ```markdown
   1. Open DocuBot
   2. Add a PDF file
   3. Try to ask a question
   4. See error
   ```
3. **Expected vs Actual Behavior**
4. **Environment**:
   - OS and version
   - Python version
   - DocuBot version
5. **Logs**: Relevant error logs
6. **Screenshots**: If applicable

### Feature Requests
Include:
1. **Problem Statement**: What problem does this solve?
2. **Proposed Solution**: How should it work?
3. **Alternatives Considered**: Other approaches
4. **Additional Context**: Any relevant information

## Feature Requests

### Feature Categories
1. **Core Features**: Document processing, AI capabilities
2. **UI/UX Improvements**: Interface enhancements
3. **Performance**: Speed, memory, optimization
4. **Integration**: Third-party tool support
5. **Accessibility**: Screen readers, keyboard navigation

### Implementation Priority
Features are prioritized based on:
1. **User Demand**: Number of requests
2. **Alignment with Vision**: Fits project goals
3. **Complexity**: Development effort required
4. **Dependencies**: External requirements

## Documentation

### Documentation Structure
```
docs/
├── user_guide/          # User documentation
├── developer_guide/     # Developer documentation
├── api_reference/       # API documentation
└── tutorials/           # Step-by-step guides
```

### Writing Documentation
- Use clear, concise language
- Include examples where helpful
- Keep documentation up to date with code
- Use Markdown formatting
- Include screenshots for UI features

### Updating Documentation
When code changes affect:
- User-facing features
- Configuration options
- API endpoints
- Installation process

Update the relevant documentation files.

## Testing

### Test Structure
```
tests/
├── unit/               # Unit tests
├── integration/        # Integration tests
├── e2e/               # End-to-end tests
└── fixtures/          # Test data
```

### Writing Tests
```python
import pytest
from src.processor import DocumentProcessor

def test_pdf_processing():
    """Test PDF document processing."""
    processor = DocumentProcessor()
    result = processor.process("test.pdf")
    
    assert result["status"] == "success"
    assert result["chunks_processed"] > 0
    assert "text" in result["content"]

def test_chunking_edge_cases():
    """Test chunking with edge cases."""
    processor = DocumentProcessor()
    
    # Test empty document
    with pytest.raises(ValueError):
        processor._chunk_text("")
    
    # Test very large document
    large_text = "x " * 10000
    chunks = processor._chunk_text(large_text)
    assert len(chunks) > 0
```

### Test Requirements
- **Unit Tests**: Cover all new functions/methods
- **Integration Tests**: Test component interactions
- **Edge Cases**: Test boundary conditions
- **Performance Tests**: For CPU/memory intensive operations

### Running Tests
```bash
# Run all tests
pytest

# Run specific test category
pytest tests/unit/
pytest tests/integration/

# Run with coverage
pytest --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/unit/test_processor.py

# Run with verbose output
pytest -v
```

## Release Process

### Versioning
We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Incompatible API changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Cycle
1. **Development**: Features developed in feature branches
2. **Alpha Testing**: Internal testing
3. **Beta Testing**: Community testing
4. **Release Candidate**: Final testing phase
5. **Stable Release**: Production release

### Release Checklist
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Changelog updated
- [ ] Version numbers updated
- [ ] Binaries built for all platforms
- [ ] Release notes prepared

## Community

### Communication Channels
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General discussion and questions
- **Discord**: Real-time chat and community support
- **Email**: arkantsabit025@gmail.com (for sensitive matters)

### Roles and Responsibilities
- **Maintainers**: Project oversight, code review, releases
- **Contributors**: Code contributions, bug fixes
- **Documenters**: Documentation improvements
- **Testers**: Testing and bug reporting
- **Translators**: Localization support

### Recognition
Contributors are recognized in:
- Release notes
- Contributors.md file
- Project documentation
- Community announcements

## Getting Help

### Development Questions
1. Check existing documentation
2. Search GitHub issues and discussions
3. Ask in Discord community
4. Create a question issue on GitHub

### Mentorship
Experienced contributors are available to mentor newcomers. If you need guidance:
1. Label your issue with `mentorship-needed`
2. Join the Discord channel for new contributors
3. Ask for help during community office hours

## License

By contributing to DocuBot, you agree that your contributions will be licensed under the project's MIT License.

## Acknowledgments

We appreciate all contributions, whether they are:
- Code improvements
- Documentation updates
- Bug reports
- Feature suggestions
- Testing and feedback
- Community support

Thank you for helping make DocuBot better!

---

*This contributing guide is adapted from many successful open-source projects. Special thanks to all contributors who help maintain and improve this document.*
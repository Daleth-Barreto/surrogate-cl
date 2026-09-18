# AI Usage Policy

## Overview

This document describes how AI language models were used in the development of the `surrogate-cl` project.

## AI tools used

- **Claude** (Anthropic): code refactoring, documentation, debugging
- **GPT-4** (OpenAI): research synthesis, literature review, test generation
- **GitHub Copilot**: code completion, function suggestions

## Permitted uses

### Code refactoring
- Extracting repeated logic into helper functions
- Improving code readability and maintainability
- Applying design patterns where appropriate
- Optimizing hot paths (with benchmarks)

### Documentation
- Writing docstrings for public APIs
- Generating inline comments for complex algorithms
- Creating and updating README files
- Writing technical documentation and guides

### Testing
- Generating unit test cases
- Identifying edge cases and boundary conditions
- Writing integration test scenarios
- Creating test fixtures and mocks

### Research
- Synthesizing related work from academic papers
- Suggesting methodological improvements
- Identifying relevant citations
- Reviewing statistical approaches

## Prohibited uses

- **Autonomous code generation**: all AI-generated code must be reviewed and tested by a human before inclusion
- **Experimental design**: the scientific methodology, experimental protocol, and interpretation of results are entirely human-authored
- **Data fabrication**: AI must not generate, fabricate, or hallucinate experimental data or results
- **Peer review**: AI must not be used to write or review peer reviews of this work

## Disclosure requirements

All contributions to this project must disclose AI usage in the pull request description. The disclosure should include:

1. Which AI tool(s) were used
2. What the tool(s) were used for
3. What percentage of the contribution was AI-assisted

Example disclosure:

> This PR uses Claude for refactoring the `task_tracking.py` module. The core logic was written by hand; Claude extracted helper functions and added docstrings.

## Attribution

AI tools are credited in the Acknowledgments section of the README. This does not imply endorsement by the AI providers.

## Version history

| Date | Change |
|------|--------|
| 2026-09-16 | Initial AI usage policy |

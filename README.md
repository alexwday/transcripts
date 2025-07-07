# Bank Transcript Database Project

## Overview
A two-stage system for creating comprehensive databases from Canadian and US bank earnings call transcripts, designed to support AI-powered financial research and chatbot applications.

## Project Structure

```
transcripts/
├── README.md                          # This file - project overview
├── CLAUDE.md                          # Context for future development sessions
├── PROJECT_STATUS.md                  # Detailed status and next steps
└── stage1-file-aggregation/           # ✅ COMPLETED
    ├── transcript_aggregator.py       # Main aggregation script
    └── README.md                      # Stage 1 documentation
```

## Stages

### Stage 1: File Aggregation ✅ COMPLETED
**Purpose**: Monitor and aggregate bank transcript files from multiple NAS locations into a centralized, organized structure.

**Status**: Production-ready (requires NAS credential configuration)

**Key Features**:
- Automated NAS monitoring with NTLM v2 authentication
- Flexible pattern matching for various naming conventions
- Smart deduplication using file timestamps
- Comprehensive error handling and logging
- Scheduled execution support

### Stage 2: LLM Processing 🔄 UPCOMING
**Purpose**: Process aggregated transcripts to create searchable databases using LLM models.

**Planned Components**:
- RAG database for semantic search
- Structured database with tagged financial metrics
- API layer for chatbot integration
- Automated content analysis pipeline

## Quick Start

### Prerequisites
```bash
pip install pysmb
```

### Configuration
Update NAS credentials in `stage1-file-aggregation/transcript_aggregator.py`:
```python
NAS_USERNAME = "your_username"
NAS_PASSWORD = "your_password"
SOURCE_NAS_IP = "192.168.1.100"  # wrkgrp30
DEST_NAS_IP = "192.168.2.100"    # wrkgrp33
```

### Running Stage 1
```bash
cd stage1-file-aggregation
python transcript_aggregator.py
```

## Data Sources
- **Canadian Banks**: Major Canadian bank earnings calls (2020+)
- **US Banks**: Major US bank earnings calls (2020+)
- **Format**: PDF transcript files
- **Frequency**: Quarterly earnings releases

## Technical Details

### Stage 1 Implementation
- **Language**: Python 3.x
- **NAS Protocol**: SMB/CIFS with NTLM v2
- **File Handling**: Streaming operations with BytesIO
- **Pattern Matching**: Flexible regex for folder naming variations
- **Scheduling**: Cron/task scheduler compatible

### Stage 2 Planning
- **PDF Processing**: Text extraction and cleaning
- **LLM Integration**: Financial text analysis and tagging
- **Vector Database**: Semantic search capabilities
- **Structured Database**: Financial metrics and time-series data
- **APIs**: RESTful endpoints for downstream applications

## Development Context

This project was developed to provide a comprehensive foundation for AI-powered financial research tools. Stage 1 handles the complex task of aggregating files from multiple sources with inconsistent naming conventions, while Stage 2 will focus on content processing and database creation.

### Key Design Decisions
- **Flexible Pattern Matching**: Handles Q1, Q121, Q1 2020, Q1 2022 variations
- **Normalized Output**: Consistent Q1-Q4 structure regardless of source naming
- **Error Resilience**: Individual file failures don't stop the entire process
- **Standalone Design**: Self-contained script with minimal dependencies

## Next Steps

1. **Deploy Stage 1**: Configure credentials and test in production environment
2. **Plan Stage 2**: Select LLM models and database technologies
3. **Design Schema**: Define metadata structure for financial content
4. **Build Pipeline**: Implement PDF processing and content analysis
5. **Create APIs**: Develop endpoints for chatbot integration

## Documentation
- Each stage has dedicated README files with detailed implementation notes
- `PROJECT_STATUS.md` provides comprehensive status tracking
- `CLAUDE.md` maintains context for future development sessions

## Support
This project is designed for financial institutions requiring automated transcript processing and analysis capabilities. The modular design allows for easy extension to additional data sources and analysis types.
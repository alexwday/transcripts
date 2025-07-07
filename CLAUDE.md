# Bank Transcript Database Project

## Project Overview
Two-stage system for creating comprehensive databases from Canadian and US bank earnings call transcripts:

1. **Stage 1**: File aggregation and organization ✅ COMPLETED
2. **Stage 2**: LLM processing and database creation 🔄 NEXT

## Stage 1: File Aggregation (COMPLETED)

### Location
`stage1-file-aggregation/` directory contains:
- `transcript_aggregator.py` - Main aggregation script
- `README.md` - Detailed stage 1 documentation

### Key Features
- **NAS Monitoring**: Automated file discovery and copying
- **Flexible Patterns**: Handles Q1, Q121, Q1 2020, Q1 2022 naming variations
- **Smart Deduplication**: Only copies new/modified files using timestamps
- **Error Resilience**: Continues processing on individual failures
- **NTLM v2 Support**: Direct TCP connection for reliable NAS access

### Configuration Status
- ✅ Flexible quarter pattern matching (Q1, Q121, Q1 2020, etc.)
- ✅ Enhanced transcript folder matching (clean final transcripts, etc.)
- ✅ Year range: 2020-2031
- ✅ NTLM v2 and direct TCP configured
- ⚠️ **TODO**: Update NAS IPs and credentials for production

### Data Flow
```
Source NAS (wrkgrp30):
├── Canadian Peer Benchmarking/
│   └── YYYY/Qxxx/[Final Transcripts]/
└── US Peer Benchmarking/
    └── YYYY/Qxxx/[Clean Transcripts]/

→ Aggregated Destination (wrkgrp33):
    database_refresh/YYYY/QX/transcript.pdf
```

## Stage 2: LLM Processing (UPCOMING)

### Planned Components
1. **PDF Processing**: Text extraction and cleaning
2. **RAG Database**: Vector embeddings for semantic search
3. **Structured Database**: Financial metrics extraction and tagging
4. **API Layer**: Chatbot integration endpoints

### Technical Stack (To Be Determined)
- PDF processing library (PyPDF2, pdfplumber, etc.)
- LLM integration (OpenAI, local models, etc.)
- Vector database (Chroma, Pinecone, etc.)
- Traditional database (PostgreSQL, SQLite, etc.)

## Dependencies
- **Stage 1**: `pysmb` for NAS connectivity
- **Stage 2**: TBD based on chosen LLM and database technologies

## Commands

### Stage 1
```bash
# Install dependencies
pip install pysmb

# Run aggregation
cd stage1-file-aggregation
python transcript_aggregator.py

# Schedule with cron
0 */4 * * * /usr/bin/python3 /path/to/transcript_aggregator.py
```

### Stage 2 (Future)
```bash
# TBD - Will be defined in next development phase
```

## Project Files
- `PROJECT_STATUS.md` - Current status and next steps
- `stage1-file-aggregation/` - Complete Stage 1 implementation
- `CLAUDE.md` - This context file for future sessions

## Development Notes

### Stage 1 Lessons Learned
- BytesIO required for pysmb file operations
- Flexible pattern matching essential for inconsistent naming
- Destination path parsing needs careful relative path handling
- NTLM v2 and direct TCP crucial for stable connections

### Stage 2 Preparation
- Stage 1 provides organized PDF files ready for processing
- Need to select LLM models suitable for financial text analysis
- Consider chunking strategy for long transcript documents
- Plan metadata schema for financial content tagging

## Session Handoff
Stage 1 is complete and production-ready after credential configuration. Next session should focus on Stage 2 planning: technology selection, architecture design, and initial implementation of PDF processing and LLM integration.
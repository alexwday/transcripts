# Bank Transcript Aggregation System

## Project Overview
This project creates a database system for retrieving information about Canadian and US bank earnings call transcripts. The system has two main stages:

1. **Stage 1 (Current)**: Standalone script that monitors NAS locations and aggregates transcript files
2. **Stage 2 (Future)**: LLM-based processing to create RAG databases and structured databases

## Current Implementation

### transcript_aggregator.py
A standalone Python script that:
- Monitors multiple NAS locations for bank earnings call transcripts
- Copies new/updated PDF files to a centralized location
- Maintains organized folder structure (Year/Quarter)
- Handles Canadian and US bank transcripts separately
- Uses file modification dates to avoid unnecessary copies
- Provides comprehensive logging and error handling

### NAS Structure
**Source Locations:**
- Canadian Banks: `wrkgrp30/Investor Relations/5. Benchmarking/Peer Benchmarking/Canadian Peer Benchmarking/`
- US Banks: `wrkgrp30/Investor Relations/5. Benchmarking/Peer Benchmarking/US Peer Benchmarking/`

**Destination:**
- `wrkgrp33/Finance Data and Analytics/DSA/AEGIS/Transcripts/database_refresh/`

**Folder Pattern:**
```
YYYY/
├── Q1/
│   └── [Final Transcripts | Clean Transcripts]/
│       ├── bank1_transcript.pdf
│       └── bank2_transcript.pdf
├── Q2/
├── Q3/
└── Q4/
```

### Configuration
All settings are hardcoded at the top of the script:
- NAS IPs and credentials
- Folder paths and naming patterns
- Year range (2001-2031)
- File extensions and target folder names

## Dependencies
- `pysmb` - For NAS/SMB connectivity
- Python 3.x standard library

## Usage
1. Update NAS IPs and credentials in script configuration
2. Install dependencies: `pip install pysmb`
3. Run script: `python transcript_aggregator.py`
4. Can be run in Jupyter notebooks or as scheduled job

## Future Development (Stage 2)
- RAG database creation from transcript content
- Structured database with tagged financial metrics
- LLM-based content processing and categorization
- API for downstream chatbot integration

## Testing
- Run script with test data to verify folder creation
- Check error logging functionality
- Verify file comparison and update logic
- Test NAS connectivity and authentication

## Commands
- **Test script**: `python transcript_aggregator.py`
- **Install dependencies**: `pip install pysmb`
- **Check logs**: Script outputs to console and creates error logs on NAS if needed
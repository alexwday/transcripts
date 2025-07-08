# Bank Transcript Database Project

## Overview
This project creates a comprehensive database system for analyzing Canadian and US bank earnings call transcripts using advanced NLP and vector search capabilities.

## Project Structure

```
transcripts/
├── stage1-file-aggregation/          # ✅ COMPLETED
│   ├── transcript_aggregator.py      # Enhanced file aggregation
│   └── README.md                     # Stage 1 documentation
├── stage2-database-processing/       # ✅ COMPLETED (Stage 2)
│   ├── file_management.py           # File comparison & DB management
│   └── README.md                    # Stage 2+ documentation
├── reference/                       # Reference implementations
│   └── transcript_processor.py      # Comprehensive processor reference
├── final_schema.sql                 # PostgreSQL database schema
├── retrieval_flow_detailed.md       # Future retrieval system architecture
├── CLAUDE.md                        # Detailed technical documentation
├── PROJECT_STATUS.md               # Comprehensive project status
└── README.md                       # This file
```

## Stages

### Stage 1: File Aggregation ✅ COMPLETED
**Location**: `stage1-file-aggregation/transcript_aggregator.py`

Automated system for collecting and organizing PDF transcript files from multiple NAS sources into a centralized, standardized structure.

**Key Features**:
- Monitors Canadian and US bank transcript folders
- Handles various naming conventions across years
- Smart deduplication based on timestamps
- **Enhanced logging** with step-by-step progress tracking
- **Comprehensive error handling** with permanent audit trails
- **Dual log system** - console output plus detailed logs in `logs/` folder
- Ready for scheduled automation

### Stage 2: File Management ✅ COMPLETED
**Location**: `stage2-database-processing/file_management.py`

Focused file comparison and master database management system.

**Key Features**:
- Checks master database existence and creates if needed
- Compares NAS files with master database records
- Identifies new, modified, and deleted files
- Outputs organized file lists for processing pipeline
- **Comprehensive logging** with execution timing
- **Ordered pipeline outputs** for subsequent stages

### Stage 3+: Database Processing 🔄 PLANNED
**Location**: `stage2-database-processing/` (future stages)

Transforms organized PDF files into searchable databases with intelligent chunking and embeddings.

**Planned Components**:
- **Stage 3**: PDF text extraction and bank identification
- **Stage 4**: Primary section classification
- **Stage 5**: Secondary section generation (chunking)
- **Stage 6**: Enhancement with summaries, embeddings, and scoring
- **Stage 7**: Database update and master database management
- **Future**: Multi-path retrieval system

## Architecture

### Data Flow
```
NAS Sources → Stage 1 → database_refresh/YYYY/QX/file.pdf → Stage 2 → refresh_outputs/ → Stages 3-7 → PostgreSQL Database
```

### Folder Structure
```
Transcripts/
├── database_refresh/     # PDF files from Stage 1 (input)
├── database/            # Master database storage
│   └── master_database.csv
├── refresh_outputs/     # Processing pipeline outputs
│   ├── 01_files_to_add.json
│   └── 02_files_to_delete.json
└── logs/               # Error and summary logs
    ├── stage1_*.log
    └── stage2_*.log
```

### Technology Stack
- **File Processing**: Python, pysmb, PyPDF2
- **Database**: PostgreSQL with pgvector extension
- **LLM Integration**: OpenAI GPT-4 and embeddings
- **Vector Search**: pgvector for similarity search
- **Infrastructure**: NAS storage, automated scheduling

## Getting Started

### Prerequisites
- Python 3.8+
- NAS access credentials
- PostgreSQL with pgvector extension (for future stages)
- OpenAI API access (for future stages)

### Stage 1 Setup
1. Update NAS credentials in `stage1-file-aggregation/transcript_aggregator.py`
2. Configure source and destination paths
3. Run: `python transcript_aggregator.py`
4. Schedule with cron for automation

### Stage 2 Setup
1. Complete Stage 1 setup
2. Update NAS credentials in `stage2-database-processing/file_management.py`
3. Run: `python file_management.py`
4. Review output files in `refresh_outputs/` folder

### Future Stages Setup
1. Complete Stages 1-2
2. Set up PostgreSQL with pgvector
3. Configure OpenAI API credentials
4. Implement Stages 3-7 following reference code in `reference/transcript_processor.py`

## Current Status

### Completed
- ✅ **Stage 1**: File aggregation system with enhanced logging
- ✅ **Stage 2**: File management and master database initialization
- ✅ **Architecture**: Modular pipeline design with clean separation
- ✅ **Logging**: Comprehensive error tracking and permanent audit trails
- ✅ **Infrastructure**: Organized folder structure and data flow

### In Progress
- 🔄 **Stage 3**: PDF text extraction and bank identification
- 🔄 **Stage 4**: Primary section classification
- 🔄 **Stage 5**: Secondary section generation (chunking)
- 🔄 **Stage 6**: Enhancement pipeline
- 🔄 **Stage 7**: Database update system

### Planned
- 📋 **PostgreSQL setup** with pgvector extension
- 📋 **Vector search and retrieval system**
- 📋 **API endpoints** for querying
- 📋 **User interface** for search and analysis

## Key Features

### Enhanced Logging System
- **Step-by-step progress tracking** with ✓ checkmarks
- **Comprehensive error collection** with try-catch blocks for each major step
- **Execution timing** and detailed statistics
- **Permanent log files** in dedicated `logs/` folder
- **Warning detection** for edge cases

### Modular Architecture
- **Focused stages** - each stage handles specific functionality
- **Ordered outputs** - clear data flow between stages
- **Clean separation** - inputs, database, outputs, and logs in separate folders
- **Reference implementations** - comprehensive processor for future development

### Production-Ready Design
- **Robust error handling** with graceful degradation
- **Comprehensive logging** for troubleshooting and monitoring
- **Scheduled execution** support with cron compatibility
- **Scalable architecture** for handling large volumes of files

## Documentation

### Core Documentation
- **`CLAUDE.md`**: Detailed technical documentation and project overview
- **`PROJECT_STATUS.md`**: Comprehensive project status and progress tracking
- **`final_schema.sql`**: PostgreSQL database schema design
- **`retrieval_flow_detailed.md`**: Future retrieval system architecture

### Stage-Specific Documentation
- **`stage1-file-aggregation/README.md`**: Stage 1 implementation details
- **`stage2-database-processing/README.md`**: Stage 2+ architecture and progress

### Reference Materials
- **`reference/transcript_processor.py`**: Comprehensive processor reference for future stages

## Contributing

This project follows a modular architecture with clear separation between stages. Each stage is designed to be:
- **Self-contained** with minimal dependencies
- **Thoroughly documented** and logged
- **Testable** and maintainable
- **Following established patterns** from previous stages

## License

Internal project - all rights reserved.
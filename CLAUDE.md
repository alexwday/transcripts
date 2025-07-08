# Bank Transcript Database Project

## Project Overview
Multi-stage system for creating comprehensive databases from Canadian and US bank earnings call transcripts:

1. **Stage 1**: File aggregation and organization ✅ COMPLETED
2. **Stage 2**: File management and database initialization ✅ COMPLETED  
3. **Stage 3+**: PDF processing and database creation 🔄 PLANNED

## Stage 1: File Aggregation (COMPLETED)

### Location
`stage1-file-aggregation/` directory contains complete implementation

### Features
- **Enhanced logging** with step-by-step progress indicators and comprehensive error tracking
- **Dual log system** - console output plus detailed logs written to `logs/` folder
- **Execution timing** and detailed statistics
- **Robust error handling** with try-catch blocks for each major step

### Data Flow
```
Source NAS → data/YYYY/QX/transcript.pdf
```

## Stage 2: File Management (COMPLETED)

### Location
`stage2-database-processing/file_management.py`

### Purpose
Focused file comparison and master database management:
- Checks if master database exists, creates if needed
- Compares NAS files with master database records
- Identifies new, modified, and deleted files
- Outputs organized file lists for processing stages

### Folder Structure
```
Transcripts/
├── data/                # PDF files from Stage 1 (input)
├── database/            # Master database storage
│   └── master_database.csv
├── database_refresh/    # Processing pipeline outputs and logs
│   ├── 01_files_to_add.json
│   ├── 02_files_to_delete.json
│   └── logs/           # Error and summary logs
│       ├── stage1_*.log
│       └── stage2_*.log
```

### Features
- **Comprehensive logging** matching Stage 1 standards
- **Execution timing** and detailed statistics
- **Error and warning tracking** with permanent log files
- **Ordered outputs** for pipeline processing

## Stage 3+: Database Processing (PLANNED)

### Overview
Will transform PDF transcripts into a searchable PostgreSQL database with:
- Hierarchical section classification (primary/secondary)
- Vector embeddings for similarity search
- Three-path retrieval system optimized for different query types

### Database Schema
```sql
transcript_sections:
├── File metadata (year, quarter, bank, ticker, filepath)
├── Hierarchical classification (primary/secondary types + summaries)
├── Content data (text, order, tokens, embeddings)
└── Scoring (importance, context relevance)
```

### Processing Pipeline
1. **Primary Section Identification**: LLM identifies 6 main sections
2. **Secondary Section Breakdown**: LLM creates contextual subsections (these ARE the chunks)
3. **Enhancement**: Generate summaries, embeddings, and relevance scores

### Retrieval System

#### Query Router
Analyzes queries to select optimal path:
- **Similarity Search**: For specific topics/metrics
- **Section Retrieval**: For section-specific queries  
- **Full Transcript**: For comprehensive analysis

#### Path 1: Similarity Search (Updated Flow)
1. Vector search → Top 20 similar sections
2. LLM relevance filtering → Remove irrelevant
3. Importance reranking → Keep top 10
4. Context enhancement → Add important neighbors
5. Section ordering → Restore transcript order
6. Gap filling → Fill small gaps under token budget
7. Synthesis → Generate answer

#### Path 2: Section Retrieval
1. Get primary summaries
2. LLM selects relevant sections
3. Retrieve complete sections
4. Synthesis

#### Path 3: Full Transcript
1. Retrieve everything
2. Synthesis

### Key Design Decisions
- No separate chunking - secondary sections are the chunks
- Early LLM filtering before expensive reranking
- Smart context inclusion based on relevance scores
- Gap filling maintains narrative continuity

### Implementation Files
- `final_schema.sql` - PostgreSQL schema design
- `retrieval_flow_detailed.md` - Complete retrieval documentation
- `stage2-database-processing/README.md` - Stage 2+ overview
- `stage2-database-processing/file_management.py` - Current Stage 2 implementation
- `reference/transcript_processor.py` - Comprehensive processor reference for future stages

### Next Steps
1. **Stage 3**: PDF text extraction and bank identification
2. **Stage 4**: Primary section classification  
3. **Stage 5**: Secondary section generation (chunking)
4. **Stage 6**: Enhancement with summaries, embeddings, and scores
5. **Stage 7**: Database update and master database management
6. **Future**: Retrieval API and synthesis layer

## Project Structure
```
transcripts/
├── stage1-file-aggregation/          # Complete file aggregation
│   ├── transcript_aggregator.py      # Enhanced with comprehensive logging
│   └── README.md
├── stage2-database-processing/       # Database creation pipeline
│   ├── file_management.py           # Current: File comparison & DB init
│   └── README.md                    # Stage 2+ overview
├── reference/                       # Reference implementations
│   └── transcript_processor.py      # Comprehensive processor reference
├── final_schema.sql                 # PostgreSQL schema design
├── retrieval_flow_detailed.md       # Future retrieval system
└── CLAUDE.md                        # This file
```
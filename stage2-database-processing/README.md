# Stage 2+: Database Processing Pipeline

## Overview

Stage 2+ encompasses the multi-stage pipeline for transforming PDF transcript files from Stage 1 into a searchable PostgreSQL database with intelligent chunking, embeddings, and a multi-path retrieval system.

## Current Implementation Status

### ✅ Stage 2: File Management (COMPLETED)
**File**: `file_management.py`

**Purpose**: File comparison and master database initialization
- Checks if master database exists, creates if needed
- Compares NAS files with master database records  
- Identifies new, modified, and deleted files
- Outputs organized file lists for processing stages

**Features**:
- Comprehensive logging with step-by-step progress indicators
- Execution timing and detailed statistics
- Error and warning tracking with permanent log files
- Ordered outputs for pipeline processing

### 🔄 Stages 3-7: Database Creation (PLANNED)
- **Stage 3**: PDF text extraction and bank identification
- **Stage 4**: Primary section classification
- **Stage 5**: Secondary section generation (chunking)
- **Stage 6**: Enhancement with summaries, embeddings, and scores
- **Stage 7**: Database update and master database management

## System Architecture

### Input
- PDF files from Stage 1: `/database_refresh/YYYY/QX/transcript.pdf`
- Files contain Canadian and US bank earnings call transcripts

### Current Output (Stage 2)
- Master database: `/database/master_database.csv`
- Processing lists: `/refresh_outputs/01_files_to_add.json`, `/refresh_outputs/02_files_to_delete.json`
- Logs: `/logs/stage2_*.log`

### Future Output (Stages 3-7)
- PostgreSQL database with hierarchical sections
- Vector embeddings for similarity search
- Three retrieval paths optimized for different query types

## Database Schema

### Core Table: `transcript_sections`

```sql
-- File metadata
fiscal_year         INTEGER      -- Year: 2020, 2021, etc.
quarter            VARCHAR(10)   -- Quarter: Q1, Q2, Q3, Q4
bank_name          VARCHAR(255)  -- Bank name: "TD Bank", "JP Morgan"
ticker_region      VARCHAR(100)  -- Combined ticker-region: "TD-CA", "JPM-US"
filepath           TEXT          -- Full NAS path to source PDF
filename           VARCHAR(255)  -- Original PDF filename
date_last_modified TIMESTAMP     -- File modification timestamp

-- Hierarchical classification
primary_section_type      VARCHAR(100)  -- Main sections (6 types)
primary_section_summary   TEXT         -- Summary of entire primary section
secondary_section_type    VARCHAR(255) -- Context-specific subsections
secondary_section_summary TEXT         -- Summary of specific chunk

-- Content and search data
section_content    TEXT          -- Full text of the secondary section
section_order      INTEGER       -- Sequential order across transcript
section_tokens     INTEGER       -- Token count for budget management
section_embedding  vector(2000)  -- OpenAI text-embedding-3-large

-- Reranking scores (0.0 to 1.0)
importance_score              DECIMAL(3,2)  -- Importance within primary section
preceding_context_relevance   DECIMAL(3,2)  -- Importance of previous section
following_context_relevance   DECIMAL(3,2)  -- Importance of next section
```

### Primary Section Types
1. Safe Harbor Statement
2. Introduction
3. Management Discussion
4. Financial Performance
5. Investor Q&A
6. Closing Remarks

## Processing Pipeline

### Phase 1: Section Identification
1. **Primary Section Detection**
   - Send transcript pages to LLM sequentially
   - LLM identifies line numbers where main sections change
   - Creates 6 standard primary sections

2. **Secondary Section Breakdown**
   - Send each primary section to LLM
   - LLM creates contextual subsections (e.g., "Capital Markets Performance")
   - These subsections become our searchable chunks

### Phase 2: Content Enhancement
1. **Summary Generation**
   - Generate summaries for primary sections
   - Generate summaries for each secondary section
   
2. **Embedding Creation**
   - Create 2000-dimension embeddings using text-embedding-3-large
   - Store for similarity search

3. **Score Calculation**
   - Calculate importance scores (0-1) for each section
   - Calculate context relevance scores for neighbors

## Retrieval System

### Query Router
Analyzes user queries and selects optimal retrieval path:
- **Similarity Search**: For specific topics/metrics
- **Section Retrieval**: For section-specific queries
- **Full Transcript**: For comprehensive analysis

### Path 1: Similarity Search (Updated Flow)

1. **Vector Similarity Search**
   - Find top 20 most similar sections using embeddings
   
2. **LLM Relevance Filtering** (NEW POSITION)
   - Send section summaries to LLM
   - Remove irrelevant sections despite semantic similarity
   
3. **Importance-Based Reranking**
   - Reorder remaining sections by importance score
   - Keep only top 10 sections
   
4. **Context Enhancement**
   - Check preceding/following context relevance scores
   - Pull in neighboring sections where relevance > 0.7
   
5. **Section Ordering**
   - Restore original transcript order
   
6. **Gap Filling**
   - Fill gaps < 5 sections if under token budget
   
7. **Synthesis**
   - Generate final answer with source attribution

### Path 2: Section Retrieval

1. **Get Primary Summaries**
   - Fetch all primary section summaries
   
2. **LLM Section Selection**
   - LLM chooses relevant primary sections
   
3. **Retrieve Full Sections**
   - Get all content from selected sections
   
4. **Synthesis**
   - Generate answer from complete sections

### Path 3: Full Transcript

1. **Retrieve Everything**
   - Get all sections in order
   
2. **Synthesis**
   - Generate comprehensive answer

## Key Innovations

1. **No Separate Chunking**: Secondary sections ARE the chunks
2. **Three-Path System**: Optimized for different query types
3. **Smart Filtering**: LLM relevance check before reranking
4. **Context Awareness**: Automatic neighbor inclusion
5. **Gap Filling**: Maintains narrative continuity

## Folder Structure
```
Transcripts/
├── database_refresh/     # PDF files from Stage 1 (input)
├── database/            # Master database storage
│   └── master_database.csv
├── refresh_outputs/     # Processing pipeline outputs
│   ├── 01_files_to_add.json    # New + modified files to process
│   └── 02_files_to_delete.json # Files to remove from master
└── logs/               # Error and summary logs
    ├── stage1_*.log
    └── stage2_*.log
```

## Implementation Status

### Completed
- ✅ **Database schema designed** (`../final_schema.sql`)
- ✅ **Retrieval flow architected** (`../retrieval_flow_detailed.md`)
- ✅ **Stage 2 file management** (`file_management.py`)
- ✅ **Modular pipeline approach** with ordered outputs
- ✅ **Comprehensive logging** and error tracking
- ✅ **Master database initialization** and management
- ✅ **Reference implementation** (`../reference/transcript_processor.py`)

### Planned (Stages 3-7)
- 🔄 **PDF processing implementation** (Stage 3)
- 🔄 **LLM section identification** (Stage 4)
- 🔄 **Secondary section generation** (Stage 5)
- 🔄 **Enhancement pipeline** (Stage 6)
- 🔄 **Database update system** (Stage 7)
- 🔄 **API development** (Future)

## Next Steps

1. **Stage 3**: Implement PDF text extraction and bank identification
2. **Stage 4**: Build LLM primary section identification pipeline
3. **Stage 5**: Create secondary section generation (chunking)
4. **Stage 6**: Implement summaries, embeddings, and scoring
5. **Stage 7**: Build database update and master database management
6. **Future**: Develop retrieval API and synthesis layer

## Configuration

Update NAS credentials and paths in each stage script before running. All configuration is at the top of each Python file following the established pattern from Stage 1.
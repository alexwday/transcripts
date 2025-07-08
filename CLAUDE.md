# Bank Transcript Database Project

## Project Overview
Two-stage system for creating comprehensive databases from Canadian and US bank earnings call transcripts:

1. **Stage 1**: File aggregation and organization ✅ COMPLETED
2. **Stage 2**: Database processing and retrieval system 🔄 IN PROGRESS

## Stage 1: File Aggregation (COMPLETED)

### Location
`stage1-file-aggregation/` directory contains complete implementation

### Data Flow
```
Source NAS → database_refresh/YYYY/QX/transcript.pdf
```

## Stage 2: Database Processing (IN PROGRESS)

### Overview
Transforms PDF transcripts into a searchable PostgreSQL database with:
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
- `stage2-database-processing/final_schema.sql` - PostgreSQL schema
- `stage2-database-processing/retrieval_flow_detailed.md` - Complete retrieval documentation
- `stage2-database-processing/README.md` - Stage 2 overview

### Next Steps
1. Implement PDF text extraction
2. Build LLM section identification
3. Create embedding generation
4. Develop retrieval API
5. Build synthesis layer

## Project Structure
```
transcripts/
├── stage1-file-aggregation/     # Complete file aggregation
├── stage2-database-processing/  # Database and retrieval system
│   ├── final_schema.sql
│   ├── retrieval_flow_detailed.md
│   └── README.md
└── CLAUDE.md                   # This file
```
-- Bank Transcript Database Schema - Final Version
-- Stage 2: Database Processing Pipeline

-- Enable pgvector extension for similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Main table storing hierarchical transcript sections
CREATE TABLE transcript_sections (
    id SERIAL PRIMARY KEY,
    
    -- File metadata
    fiscal_year INTEGER NOT NULL,
    quarter VARCHAR(10) NOT NULL,
    bank_name VARCHAR(255) NOT NULL,
    ticker_region VARCHAR(100) NOT NULL,
    filepath TEXT NOT NULL,
    filename VARCHAR(255) NOT NULL,
    date_last_modified TIMESTAMP NOT NULL,
    
    -- Hierarchical classification
    primary_section_type VARCHAR(100) NOT NULL,
    primary_section_summary TEXT,
    secondary_section_type VARCHAR(255) NOT NULL,
    secondary_section_summary TEXT,
    
    -- Content and search data
    section_content TEXT NOT NULL,
    section_order INTEGER NOT NULL,
    section_tokens INTEGER NOT NULL,
    section_embedding vector(2000),
    
    -- Reranking scores
    importance_score DECIMAL(3,2),
    preceding_context_relevance DECIMAL(3,2),
    following_context_relevance DECIMAL(3,2),
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(filepath, section_order)
);

-- Performance indexes
CREATE INDEX idx_transcript_sections_embedding ON transcript_sections 
USING ivfflat (section_embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_transcript_sections_filepath ON transcript_sections(filepath);
CREATE INDEX idx_transcript_sections_bank_quarter ON transcript_sections(bank_name, fiscal_year, quarter);
CREATE INDEX idx_transcript_sections_importance ON transcript_sections(importance_score);

-- Update trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_transcript_sections_updated_at 
    BEFORE UPDATE ON transcript_sections 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
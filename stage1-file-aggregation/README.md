# Stage 1: File Aggregation

## Overview
This stage handles the automated monitoring and aggregation of bank earnings call transcripts from multiple NAS locations into a centralized, organized structure.

## Purpose
- Monitor Canadian and US bank transcript folders on NAS
- Copy new/updated PDF files to centralized location
- Maintain consistent folder structure (Year/Quarter)
- Handle various naming conventions across different years
- Provide foundation for Stage 2 processing

## Key Features

### File Monitoring
- **Sources**: Canadian and US bank transcript folders on wrkgrp30 share
- **Destination**: Centralized location on wrkgrp33 share
- **File Types**: PDF transcript files only
- **Scheduling**: Designed to run on schedule (cron/task scheduler)

### Flexible Pattern Matching
- **Years**: 2020-2031 (configurable range)
- **Quarter Folders**: Handles Q1, Q121, Q1 2020, Q1 2022, etc.
- **Transcript Folders**: Matches "final transcripts", "clean transcripts", "clean final transcripts", etc.
- **Case Insensitive**: All pattern matching ignores case

### Smart File Management
- **Deduplication**: Only copies new or modified files
- **Date Comparison**: Uses last modified timestamps
- **Folder Creation**: Automatically creates year/quarter structure
- **Error Handling**: Continues processing on individual file failures

## Files

### `transcript_aggregator.py`
Main script that performs the file aggregation. Features:
- Hardcoded configuration at top for easy modification
- NTLM v2 authentication with direct TCP
- Comprehensive logging and error reporting
- Can be run standalone or in Jupyter notebooks

## Configuration

### NAS Settings
```python
# Update these in transcript_aggregator.py
NAS_USERNAME = "your_username"
NAS_PASSWORD = "your_password"
SOURCE_NAS_IP = "192.168.1.100"  # wrkgrp30 location
DEST_NAS_IP = "192.168.2.100"    # wrkgrp33 location
```

### Folder Structure
```
Source (wrkgrp30):
├── Canadian Peer Benchmarking/
│   ├── 2024/
│   │   ├── Q1 2024/                    # Year-specific naming
│   │   │   └── Final Transcripts/
│   │   └── Q2 2024/
│   └── 2025/
│       └── Q125/                       # Year-specific naming
└── US Peer Benchmarking/
    ├── 2024/
    │   ├── Q1/                         # Standard naming
    │   │   └── Clean Transcripts/
    │   └── Q2/
    └── 2025/

Destination (wrkgrp33):
Finance Data and Analytics/DSA/AEGIS/Transcripts/database_refresh/
├── 2024/
│   ├── Q1/                             # Normalized naming
│   │   ├── bank1_q1_2024.pdf
│   │   └── bank2_q1_2024.pdf
│   └── Q2/
└── 2025/
```

## Usage

### Prerequisites
```bash
pip install pysmb
```

### Running the Script
```bash
# Standalone execution
python transcript_aggregator.py

# Or in Jupyter notebook
exec(open('transcript_aggregator.py').read())
```

### Scheduling
Add to cron for automated execution:
```bash
# Run every 4 hours
0 */4 * * * /usr/bin/python3 /path/to/transcript_aggregator.py
```

## Logging Output
The script provides meaningful progress information:
- Connection status to each NAS
- File counts from each source location
- Directory creation messages
- File copy progress with [current/total] counters
- Summary with new/updated/failed counts
- Error log creation on NAS if failures occur

## Error Handling
- Individual file failures don't stop the entire process
- Detailed error messages for troubleshooting
- Error log written to NAS base directory if failures occur
- Retry-friendly design for network issues

## Testing
- Update NAS IPs and credentials in configuration
- Run script in test mode first to verify connections
- Check logs for successful file discovery and copying
- Verify folder structure creation in destination

## Performance Notes
- Processes files as it finds them (not batch)
- Typical 30-page PDF files copy quickly
- Network latency is main performance factor
- Memory usage is minimal (streaming file operations)

## Next Steps
Stage 1 provides the organized file foundation for Stage 2, which will:
- Process transcript content with LLM models
- Create RAG databases for semantic search
- Build structured databases with tagged financial metrics
- Provide APIs for downstream applications
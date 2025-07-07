#!/usr/bin/env python3
"""
Bank Earnings Call Transcript Aggregator
Monitors NAS locations for new transcript files and copies them to a centralized location
"""

import os
import sys
import time
import logging
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import hashlib

# pysmb imports
try:
    from smb.SMBConnection import SMBConnection
    from smb.smb_structs import OperationFailure
except ImportError:
    print("ERROR: pysmb not installed. Please run: pip install pysmb")
    sys.exit(1)

# ========================================
# CONFIGURATION - MODIFY AS NEEDED
# ========================================

# NAS Authentication (same for all connections)
NAS_USERNAME = "your_username"
NAS_PASSWORD = "your_password"
CLIENT_MACHINE_NAME = "PYTHON_SCRIPT"  # Can be any name
SERVER_MACHINE_NAME = "NAS_SERVER"     # Can be any name

# Source NAS Configuration
SOURCE_NAS_IP = "192.168.1.100"  # Replace with actual IP
SOURCE_NAS_PORT = 445

# Canadian Banks Source
CANADIAN_CONFIG = {
    "name": "Canadian Banks",
    "share": "wrkgrp30",
    "base_path": "Investor Relations/5. Benchmarking/Peer Benchmarking/Canadian Peer Benchmarking"
}

# US Banks Source  
US_CONFIG = {
    "name": "US Banks",
    "share": "wrkgrp30",
    "base_path": "Investor Relations/5. Benchmarking/Peer Benchmarking/US Peer Benchmarking"
}

# Destination NAS Configuration
DEST_NAS_IP = "192.168.2.100"  # Replace with actual IP
DEST_NAS_PORT = 445
DEST_CONFIG = {
    "share": "wrkgrp33",
    "base_path": "Finance Data and Analytics/DSA/AEGIS/Transcripts/database_refresh"
}

# Processing Configuration
VALID_YEAR_RANGE = (2020, 2031)  # Updated to start from 2020
FILE_EXTENSIONS = [".pdf", ".PDF"]

# ========================================
# FLEXIBLE PATTERN MATCHING
# ========================================
# NOTE: These patterns handle various naming conventions using regex-like matching
# Patterns are case-insensitive and designed to catch variations

# Quarter folder patterns (case-insensitive)
# Matches: Q1, Q1xx, Q1 xxxx, Q121, Q1 2020, etc.
QUARTER_PATTERNS = [
    r"^Q1",  # Matches anything starting with Q1
    r"^Q2",  # Matches anything starting with Q2  
    r"^Q3",  # Matches anything starting with Q3
    r"^Q4"   # Matches anything starting with Q4
]

# Target transcript folder patterns (case-insensitive)
# Matches various combinations of: final, clean, transcript(s)
TRANSCRIPT_FOLDER_PATTERNS = [
    r"final.*transcript",      # Matches "final transcripts", "final transcript", etc.
    r"clean.*transcript",      # Matches "clean transcripts", "clean transcript", etc.
    r"clean.*final.*transcript"  # Matches "clean final transcripts", etc.
]

# ========================================
# LOGGING SETUP
# ========================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ========================================
# HELPER FUNCTIONS
# ========================================

def create_smb_connection(server_ip: str, username: str, password: str, port: int = 445) -> SMBConnection:
    """Create and return an SMB connection"""
    try:
        conn = SMBConnection(username, password, CLIENT_MACHINE_NAME, SERVER_MACHINE_NAME, 
                           use_ntlm_v2=True, is_direct_tcp=True)
        connected = conn.connect(server_ip, port=port)
        if connected:
            logger.info(f"Successfully connected to {server_ip}:{port}")
            return conn
        else:
            raise Exception(f"Failed to connect to {server_ip}:{port}")
    except Exception as e:
        logger.error(f"Connection error to {server_ip}:{port}: {str(e)}")
        raise

def is_valid_year_folder(folder_name: str) -> bool:
    """Check if folder name is a valid fiscal year"""
    try:
        year = int(folder_name)
        return VALID_YEAR_RANGE[0] <= year <= VALID_YEAR_RANGE[1]
    except ValueError:
        return False

def is_quarter_folder(folder_name: str) -> Tuple[bool, str]:
    """Check if folder matches any quarter pattern and return the normalized quarter"""
    import re
    folder_lower = folder_name.lower()
    
    for pattern in QUARTER_PATTERNS:
        if re.match(pattern, folder_lower):
            # Extract the quarter number and normalize it
            if folder_lower.startswith('q1'):
                return True, 'Q1'
            elif folder_lower.startswith('q2'):
                return True, 'Q2'
            elif folder_lower.startswith('q3'):
                return True, 'Q3'
            elif folder_lower.startswith('q4'):
                return True, 'Q4'
    
    return False, ''

def is_target_folder(folder_name: str) -> bool:
    """Check if folder matches any of the target transcript patterns (case-insensitive)"""
    import re
    folder_lower = folder_name.lower()
    
    for pattern in TRANSCRIPT_FOLDER_PATTERNS:
        if re.search(pattern, folder_lower):
            return True
    
    return False

def get_file_info(conn: SMBConnection, share: str, file_path: str) -> Dict:
    """Get file information including last modified time"""
    try:
        attributes = conn.getAttributes(share, file_path)
        return {
            'path': file_path,
            'last_modified': attributes.last_write_time,
            'size': attributes.file_size
        }
    except Exception as e:
        logger.error(f"Error getting file info for {file_path}: {str(e)}")
        return None

def scan_destination(conn: SMBConnection, share: str, base_path: str) -> Dict[str, Dict]:
    """Scan destination folder and catalog existing files"""
    logger.info("Scanning destination folder for existing files...")
    existing_files = {}
    files_count = 0
    
    try:
        # Recursively scan the destination
        def scan_folder(path):
            nonlocal files_count
            try:
                items = conn.listPath(share, path)
                for item in items:
                    if item.filename in ['.', '..']:
                        continue
                    
                    item_path = f"{path}/{item.filename}" if path else item.filename
                    
                    if item.isDirectory:
                        scan_folder(item_path)
                    else:
                        if any(item.filename.endswith(ext) for ext in FILE_EXTENSIONS):
                            # Create a standardized key: year/quarter/filename
                            path_parts = item_path.split('/')
                            if len(path_parts) >= 3:
                                year = None
                                quarter = None
                                
                                # Find year and quarter in path
                                for part in path_parts:
                                    if is_valid_year_folder(part):
                                        year = part
                                    else:
                                        is_quarter, normalized_quarter = is_quarter_folder(part)
                                        if is_quarter:
                                            quarter = normalized_quarter
                                
                                if year and quarter:
                                    key = f"{year}/{quarter}/{item.filename}"
                                    existing_files[key] = {
                                        'path': item_path,
                                        'last_modified': item.last_write_time,
                                        'size': item.file_size
                                    }
                                    files_count += 1
            except Exception as e:
                logger.error(f"Error scanning folder {path}: {str(e)}")
        
        scan_folder(base_path)
        logger.info(f"Found {files_count} existing transcript files in destination")
        
    except Exception as e:
        logger.error(f"Error scanning destination: {str(e)}")
    
    return existing_files

def scan_source(conn: SMBConnection, config: Dict) -> Dict[str, Dict]:
    """Scan source folder for transcript files"""
    logger.info(f"Scanning {config['name']} source folder...")
    source_files = {}
    files_count = 0
    
    try:
        base_items = conn.listPath(config['share'], config['base_path'])
        
        for year_item in base_items:
            if year_item.filename in ['.', '..'] or not year_item.isDirectory:
                continue
            
            if not is_valid_year_folder(year_item.filename):
                continue
            
            year = year_item.filename
            year_path = f"{config['base_path']}/{year}"
            
            try:
                quarter_items = conn.listPath(config['share'], year_path)
                
                for quarter_item in quarter_items:
                    if quarter_item.filename in ['.', '..'] or not quarter_item.isDirectory:
                        continue
                    
                    is_quarter, normalized_quarter = is_quarter_folder(quarter_item.filename)
                    if not is_quarter:
                        continue
                    
                    quarter = normalized_quarter  # Use normalized quarter (Q1, Q2, etc.)
                    quarter_path = f"{year_path}/{quarter_item.filename}"  # Use actual folder name for path
                    
                    try:
                        transcript_folders = conn.listPath(config['share'], quarter_path)
                        
                        for folder_item in transcript_folders:
                            if folder_item.filename in ['.', '..'] or not folder_item.isDirectory:
                                continue
                            
                            if is_target_folder(folder_item.filename):
                                transcript_path = f"{quarter_path}/{folder_item.filename}"
                                
                                try:
                                    files = conn.listPath(config['share'], transcript_path)
                                    
                                    for file_item in files:
                                        if file_item.filename in ['.', '..'] or file_item.isDirectory:
                                            continue
                                        
                                        if any(file_item.filename.endswith(ext) for ext in FILE_EXTENSIONS):
                                            file_path = f"{transcript_path}/{file_item.filename}"
                                            key = f"{year}/{quarter}/{file_item.filename}"
                                            
                                            source_files[key] = {
                                                'path': file_path,
                                                'share': config['share'],
                                                'last_modified': file_item.last_write_time,
                                                'size': file_item.file_size,
                                                'source': config['name']
                                            }
                                            files_count += 1
                                            
                                except Exception as e:
                                    logger.error(f"Error listing transcript files in {transcript_path}: {str(e)}")
                                    
                    except Exception as e:
                        logger.error(f"Error listing folders in {quarter_path}: {str(e)}")
                        
            except Exception as e:
                logger.error(f"Error listing quarters in {year_path}: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error scanning {config['name']} source: {str(e)}")
    
    logger.info(f"Found {files_count} transcript files in {config['name']} source")
    return source_files

def ensure_directory_exists(conn: SMBConnection, share: str, path: str):
    """Ensure directory exists, create if necessary"""
    parts = path.split('/')
    current_path = ""
    
    for part in parts:
        if not part:
            continue
            
        if current_path:
            current_path = f"{current_path}/{part}"
        else:
            current_path = part
        
        try:
            # Check if directory exists
            conn.listPath(share, current_path)
        except OperationFailure:
            # Directory doesn't exist, create it
            try:
                conn.createDirectory(share, current_path)
                logger.info(f"Created directory: {current_path}")
            except Exception as e:
                logger.error(f"Failed to create directory {current_path}: {str(e)}")
                raise

def copy_file(source_conn: SMBConnection, dest_conn: SMBConnection, 
              source_share: str, dest_share: str, source_path: str, dest_path: str) -> bool:
    """Copy file from source to destination"""
    try:
        # Create temporary file-like object to hold the data
        from io import BytesIO
        temp_file = BytesIO()
        
        # Read file from source
        source_conn.retrieveFile(source_share, source_path, temp_file)
        
        # Ensure destination directory exists
        dest_dir = '/'.join(dest_path.split('/')[:-1])
        ensure_directory_exists(dest_conn, dest_share, dest_dir)
        
        # Reset file pointer to beginning
        temp_file.seek(0)
        
        # Write file to destination
        dest_conn.storeFile(dest_share, dest_path, temp_file)
        
        # Clean up
        temp_file.close()
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to copy {source_path} to {dest_path}: {str(e)}")
        return False

def write_error_log(dest_conn: SMBConnection, errors: List[str]):
    """Write error log to NAS"""
    if not errors:
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"transcript_sync_errors_{timestamp}.log"
    log_path = f"{DEST_CONFIG['base_path']}/{log_filename}"
    
    log_content = f"Transcript Sync Error Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    log_content += "=" * 60 + "\n\n"
    
    for error in errors:
        log_content += f"{error}\n"
    
    try:
        from io import BytesIO
        log_file = BytesIO(log_content.encode('utf-8'))
        dest_conn.storeFile(DEST_CONFIG['share'], log_path, log_file)
        log_file.close()
        logger.info(f"Error log written to: {log_path}")
    except Exception as e:
        logger.error(f"Failed to write error log: {str(e)}")

def main():
    """Main execution function"""
    logger.info("Starting Bank Transcript Aggregator")
    logger.info("=" * 60)
    
    errors = []
    stats = {
        'new_files': 0,
        'updated_files': 0,
        'failed_copies': 0,
        'total_processed': 0
    }
    
    source_conn = None
    dest_conn = None
    
    try:
        # Connect to source NAS
        logger.info(f"Connecting to source NAS ({SOURCE_NAS_IP})...")
        source_conn = create_smb_connection(SOURCE_NAS_IP, NAS_USERNAME, NAS_PASSWORD, SOURCE_NAS_PORT)
        
        # Connect to destination NAS
        logger.info(f"Connecting to destination NAS ({DEST_NAS_IP})...")
        dest_conn = create_smb_connection(DEST_NAS_IP, NAS_USERNAME, NAS_PASSWORD, DEST_NAS_PORT)
        
        # Scan destination for existing files
        existing_files = scan_destination(dest_conn, DEST_CONFIG['share'], DEST_CONFIG['base_path'])
        
        # Scan both sources
        all_source_files = {}
        
        # Canadian banks
        canadian_files = scan_source(source_conn, CANADIAN_CONFIG)
        all_source_files.update(canadian_files)
        
        # US banks
        us_files = scan_source(source_conn, US_CONFIG)
        all_source_files.update(us_files)
        
        logger.info(f"\nTotal source files found: {len(all_source_files)}")
        logger.info(f"Existing destination files: {len(existing_files)}")
        
        # Compare and process files
        files_to_copy = []
        
        for key, source_info in all_source_files.items():
            if key not in existing_files:
                # New file
                files_to_copy.append((key, source_info, 'new'))
                stats['new_files'] += 1
            elif source_info['last_modified'] > existing_files[key]['last_modified']:
                # Updated file
                files_to_copy.append((key, source_info, 'updated'))
                stats['updated_files'] += 1
        
        logger.info(f"\nFiles to process: {len(files_to_copy)}")
        logger.info(f"  - New files: {stats['new_files']}")
        logger.info(f"  - Updated files: {stats['updated_files']}")
        
        # Copy files
        if files_to_copy:
            logger.info("\nStarting file copy process...")
            
            for i, (key, source_info, status) in enumerate(files_to_copy, 1):
                year, quarter, filename = key.split('/')
                dest_path = f"{DEST_CONFIG['base_path']}/{year}/{quarter}/{filename}"
                
                logger.info(f"[{i}/{len(files_to_copy)}] Copying {status} file: {filename} ({source_info['source']})")
                
                success = copy_file(
                    source_conn, dest_conn,
                    source_info['share'], DEST_CONFIG['share'],
                    source_info['path'], dest_path
                )
                
                if success:
                    stats['total_processed'] += 1
                else:
                    stats['failed_copies'] += 1
                    error_msg = f"Failed to copy: {source_info['path']} -> {dest_path}"
                    errors.append(error_msg)
        else:
            logger.info("\nNo files to copy - destination is up to date")
        
        # Write error log if needed
        if errors:
            write_error_log(dest_conn, errors)
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("SYNC COMPLETE - Summary:")
        logger.info(f"  - New files copied: {stats['new_files'] - (stats['failed_copies'] if stats['new_files'] > 0 else 0)}")
        logger.info(f"  - Updated files copied: {stats['updated_files'] - (stats['failed_copies'] if stats['updated_files'] > 0 else 0)}")
        logger.info(f"  - Failed copies: {stats['failed_copies']}")
        logger.info(f"  - Total processed successfully: {stats['total_processed']}")
        
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        errors.append(f"Critical error: {str(e)}")
        
    finally:
        # Clean up connections
        if source_conn:
            source_conn.close()
        if dest_conn:
            dest_conn.close()
        
        logger.info("\nScript execution completed")

if __name__ == "__main__":
    main()
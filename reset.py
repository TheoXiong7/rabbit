#!/usr/bin/env python3
"""
Reset script for clearing logs, optreps, and database for new paper account testing.
"""

import os
import shutil
import sqlite3
from pathlib import Path


def clear_directory(directory_path):
    """Clear all contents of a directory while keeping the directory itself."""
    if os.path.exists(directory_path):
        for item in os.listdir(directory_path):
            item_path = os.path.join(directory_path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
                print(f"Removed file: {item_path}")
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
                print(f"Removed directory: {item_path}")
    else:
        print(f"Directory does not exist: {directory_path}")


def clear_database(db_path):
    """Clear the trading database by removing all tables."""
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get all table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            # Drop all tables
            for table in tables:
                table_name = table[0]
                cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                print(f"Dropped table: {table_name}")
            
            conn.commit()
            conn.close()
            print(f"Database cleared: {db_path}")
            
        except Exception as e:
            print(f"Error clearing database: {e}")
    else:
        print(f"Database does not exist: {db_path}")


def main():
    """Main reset function."""
    script_dir = Path(__file__).parent
    
    # Define paths
    logs_dir = script_dir / "logs"
    optreps_dir = script_dir / "optreps"
    db_path = script_dir / "trading_performance.db"
    
    print("Starting reset process...")
    print("=" * 50)
    
    # Clear logs directory
    print("\n1. Clearing logs directory...")
    clear_directory(logs_dir)
    
    # Clear optreps directory
    print("\n2. Clearing optreps directory...")
    clear_directory(optreps_dir)
    
    # Clear database
    print("\n3. Clearing database...")
    clear_database(db_path)
    
    print("\n" + "=" * 50)
    print("Reset process completed!")
    print("Ready for new paper account testing.")


if __name__ == "__main__":
    main()
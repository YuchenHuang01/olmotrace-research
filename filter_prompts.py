#!/usr/bin/env python3
"""
Remove crisis_alarmist entries from prompts.csv
"""

import pandas as pd
import sys

def filter_crisis_alarmist(input_csv, output_csv):
    """Remove crisis_alarmist rows from prompts CSV."""
    
    # Read the CSV
    df = pd.read_csv(input_csv)
    
    print(f"Original CSV: {len(df)} rows")
    print(f"Bucket distribution:")
    print(df['bucket'].value_counts())
    
    # Filter out crisis_alarmist
    filtered_df = df[df['bucket'] != 'crisis_alarmist']
    
    print(f"\nFiltered CSV: {len(filtered_df)} rows")
    print(f"Removed: {len(df) - len(filtered_df)} crisis_alarmist rows")
    print(f"New bucket distribution:")
    print(filtered_df['bucket'].value_counts())
    
    # Save filtered CSV
    filtered_df.to_csv(output_csv, index=False)
    print(f"\nSaved filtered CSV to: {output_csv}")
    
    return filtered_df

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python filter_prompts.py input.csv output.csv")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    filter_crisis_alarmist(input_file, output_file)
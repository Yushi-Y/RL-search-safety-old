#!/usr/bin/env python3
"""
Script to combine arditi_harmful_train.json and arditi_harmful_val_question.json 
into a single arditi_harmful_full.json file.
"""

import json
import os

def combine_json_files():
    # File paths
    train_file = "refusal_datasets/arditi_harmful_train.json"
    val_file = "refusal_datasets/arditi_harmful_val_question.json"
    output_file = "refusal_datasets/arditi_harmful_full.json"
    
    # Check if input files exist
    if not os.path.exists(train_file):
        print(f"Error: {train_file} not found")
        return
    
    if not os.path.exists(val_file):
        print(f"Error: {val_file} not found")
        return
    
    try:
        # Read training data
        print(f"Reading {train_file}...")
        with open(train_file, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        print(f"Loaded {len(train_data)} entries from training file")
        
        # Read validation data
        print(f"Reading {val_file}...")
        with open(val_file, 'r', encoding='utf-8') as f:
            val_data = json.load(f)
        print(f"Loaded {len(val_data)} entries from validation file")
        
        # Combine the data
        combined_data = train_data + val_data
        print(f"Combined data contains {len(combined_data)} total entries")
        
        # Write combined data to output file
        print(f"Writing combined data to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=4, ensure_ascii=False)
        
        print(f"Successfully created {output_file}")
        print(f"Total entries: {len(combined_data)}")
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    combine_json_files() 
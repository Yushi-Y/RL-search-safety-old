import json
import re

def clean_json_content(content):
    """Clean malformed JSON content with minimal edits to preserve original responses"""
    # Only remove the specific problematic characters that break JSON parsing
    # Remove the stray 'å' character that was causing the error
    content = content.replace('å', '')
    
    # Fix the specific malformed line that was causing the parsing error
    # Line 11 had: "  },å" - remove the stray character after comma
    content = re.sub(r',\s*[^\w\s,{}[\]":\n\r\t]+(?=\s*[}\]])', ',', content)
    
    # Only fix the most critical JSON syntax issues, don't modify response content
    return content

def load_json_with_cleaning(file_path):
    """Load JSON file with automatic cleaning if needed"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully loaded {file_path} with {len(data)} items")
        return data
    except json.JSONDecodeError as e:
        print(f"JSON parsing error in {file_path}: {e}")
        print("Attempting to clean the file...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Clean the content
        cleaned_content = clean_json_content(content)
        
        try:
            data = json.loads(cleaned_content)
            print(f"Successfully loaded cleaned {file_path} with {len(data)} items")
            return data
        except json.JSONDecodeError as e2:
            print(f"Failed to parse even after cleaning: {e2}")
            print("Attempting to extract valid JSON entries manually...")
            
            # Try to extract valid JSON entries by finding complete objects
            try:
                # Look for complete JSON objects
                json_objects = []
                lines = cleaned_content.split('\n')
                current_object = []
                brace_count = 0
                in_object = False
                
                for line in lines:
                    if '{' in line and not in_object:
                        in_object = True
                        current_object = [line]
                        brace_count = line.count('{') - line.count('}')
                    elif in_object:
                        current_object.append(line)
                        brace_count += line.count('{') - line.count('}')
                        
                        if brace_count == 0:
                            # Complete object found
                            object_str = '\n'.join(current_object)
                            try:
                                obj = json.loads(object_str)
                                json_objects.append(obj)
                            except:
                                pass  # Skip invalid objects
                            in_object = False
                            current_object = []
                
                if json_objects:
                    print(f"Successfully extracted {len(json_objects)} valid JSON objects")
                    return json_objects
                else:
                    raise Exception("Could not extract any valid JSON objects")
                    
            except Exception as extract_error:
                print(f"Failed to extract valid objects: {extract_error}")
                raise

def main():
    # Load the IT responses
    it_file = "refusal_datasets/arditi_harmful_responses_it.json"
    print(f"Loading IT responses from {it_file}...")
    it_data = load_json_with_cleaning(it_file)
    
    # Load the search responses
    search_file = "refusal_datasets/arditi_harmful_responses_search.json"
    print(f"Loading search responses from {search_file}...")
    search_data = load_json_with_cleaning(search_file)
    
    print(f"Found {len(it_data)} IT responses and {len(search_data)} search responses")
    
    # Create a mapping of question_index to responses for easier lookup
    it_map = {}
    for item in it_data:
        question_index = item.get("question_index")
        if question_index is not None:
            it_map[question_index] = item
    
    search_map = {}
    for item in search_data:
        question_index = item.get("question_index")
        if question_index is not None:
            search_map[question_index] = item
    
    # Merge the responses by question_index
    merged_data = []
    all_indices = set(it_map.keys()) | set(search_map.keys())
    
    print(f"Found {len(all_indices)} unique question indices to merge")
    
    for question_index in sorted(all_indices):
        it_item = it_map.get(question_index, {})
        search_item = search_map.get(question_index, {})
        
        merged_item = {
            "question": it_item.get("question", search_item.get("question", "")),
            "response_it": it_item.get("response", ""),
            "response_search": search_item.get("response", ""),
            "question_index": question_index
        }
        
        merged_data.append(merged_item)
        
        if len(merged_data) % 100 == 0:
            print(f"Processed {len(merged_data)}/{len(all_indices)} questions")
    
    # Save the merged data
    output_file = "refusal_datasets/arditi_harmful_responses_merged.json"
    print(f"\nSaving merged data to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)
    
    print(f"Merge complete! Saved {len(merged_data)} merged responses to {output_file}")
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"- Questions with both: {len(set(it_map.keys()) & set(search_map.keys()))}")
    print(f"- Questions with only IT: {len(set(it_map.keys()) - set(search_map.keys()))}")
    print(f"- Questions with only search: {len(set(search_map.keys()) - set(it_map.keys()))}")

if __name__ == "__main__":
    main() 
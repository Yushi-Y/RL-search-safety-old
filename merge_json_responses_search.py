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
    # Load the imperative search responses
    imperative_file = "refusal_responses/arditi_refusal_full_imperative_search.json"
    print(f"Loading imperative search responses from {imperative_file}...")
    imperative_data = load_json_with_cleaning(imperative_file)
    
    # Load the questions search responses
    questions_file = "refusal_responses/arditi_refusal_full_questions_search.json"
    print(f"Loading questions search responses from {questions_file}...")
    questions_data = load_json_with_cleaning(questions_file)
    
    print(f"Found {len(imperative_data)} imperative responses and {len(questions_data)} questions responses")
    
    # Create a mapping of question_index to responses for easier lookup
    imperative_map = {}
    for item in imperative_data:
        question_index = item.get("question_index")
        if question_index is not None:
            imperative_map[question_index] = item
    
    questions_map = {}
    for item in questions_data:
        question_index = item.get("question_index")
        if question_index is not None:
            questions_map[question_index] = item
    
    # Merge the responses by question_index
    merged_data = []
    all_indices = set(imperative_map.keys()) | set(questions_map.keys())
    
    print(f"Found {len(all_indices)} unique question indices to merge")
    
    for question_index in sorted(all_indices):
        imperative_item = imperative_map.get(question_index, {})
        questions_item = questions_map.get(question_index, {})
        
        merged_item = {
            "question_imperative": imperative_item.get("question", ""),
            "response_imperative": imperative_item.get("response", ""),
            "search_information_imperative": imperative_item.get("search_information", []),
            "question_questions": questions_item.get("question", ""),
            "response_questions": questions_item.get("response", ""),
            "search_information_questions": questions_item.get("search_information", []),
            "question_index": question_index
        }
        
        merged_data.append(merged_item)
        
        if len(merged_data) % 100 == 0:
            print(f"Processed {len(merged_data)}/{len(all_indices)} questions")
    
    # Save the merged data
    output_file = "refusal_responses/arditi_refusal_full_search_merged.json"
    print(f"\nSaving merged data to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)
    
    print(f"Merge complete! Saved {len(merged_data)} merged responses to {output_file}")
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"- Questions with both imperative and questions: {len(set(imperative_map.keys()) & set(questions_map.keys()))}")
    print(f"- Questions with only imperative: {len(set(imperative_map.keys()) - set(questions_map.keys()))}")
    print(f"- Questions with only questions: {len(set(questions_map.keys()) - set(imperative_map.keys()))}")
    
    # Additional statistics about search information
    imperative_with_search = sum(1 for item in imperative_data if item.get("search_information", []))
    questions_with_search = sum(1 for item in questions_data if item.get("search_information", []))
    print(f"- Imperative responses with search information: {imperative_with_search}/{len(imperative_data)} ({imperative_with_search/len(imperative_data)*100:.1f}%)")
    print(f"- Questions responses with search information: {questions_with_search}/{len(questions_data)} ({questions_with_search/len(questions_data)*100:.1f}%)")

if __name__ == "__main__":
    main() 
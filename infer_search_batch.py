import transformers
import torch
import random
from datasets import load_dataset
import requests
import json
import time

# Model ID and device setup
model_id = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-ppo"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

curr_eos = [151645, 151643] # for Qwen2.5 series models
curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'

# Initialize the tokenizer and model
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
# model = transformers.AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    attn_implementation="eager"  # Disable flash attention
)

# Define the custom stopping criterion
class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        # Encode the string so we have the exact token-IDs pattern
        self.target_ids = [tokenizer.encode(target_sequence, add_special_tokens=False) for target_sequence in target_sequences]
        self.target_lengths = [len(target_id) for target_id in self.target_ids]
        self._tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # Make sure the target IDs are on the same device
        targets = [torch.as_tensor(target_id, device=input_ids.device) for target_id in self.target_ids]

        if input_ids.shape[1] < min(self.target_lengths):
            return False

        # Compare the tail of input_ids with our target_ids
        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                return True

        return False

def get_query(text):
    import re
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1]
    else:
        return None

def search(query: str):
    payload = {
            "queries": [query],
            "topk": 3,
            "return_scores": True
        }
    results = requests.post("http://127.0.0.1:8000/retrieve", json=payload).json()['result']
                
    def _passages2string(retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
                        
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        return format_reference

    return _passages2string(results[0])

# Initialize the stopping criteria
target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, tokenizer)])

def process_single_question(question_text):
    """Process a single question using the same logic as infer_search.py"""
    
    question = question_text.strip()
    if question[-1] != '?':
        question += '?'
    
    # Prepare the message - exactly the same as infer_search.py
    prompt = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""

    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)

    print(f'\n\n################# [Start Reasoning + Searching] ##################\n\n')
    print(f"Question: {question}")
    
    # Initialize variables exactly like infer_search.py
    cnt = 0
    full_response = ""
    current_prompt = prompt
    
    # Process the question with potential search iterations - same logic as infer_search.py
    while True:
        input_ids = tokenizer.encode(current_prompt, return_tensors='pt').to(device)
        attention_mask = torch.ones_like(input_ids)
        
        # Generate text with the stopping criteria
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1024,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7
        )

        if outputs[0][-1].item() in curr_eos:
            generated_tokens = outputs[0][input_ids.shape[1]:]
            output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            full_response += output_text
            print(output_text)
            break

        generated_tokens = outputs[0][input_ids.shape[1]:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        full_response += output_text
        
        tmp_query = get_query(tokenizer.decode(outputs[0], skip_special_tokens=True))
        if tmp_query:
            # print(f'searching "{tmp_query}"...')
            search_results = search(tmp_query)
        else:
            search_results = ''

        search_text = curr_search_template.format(output_text=output_text, search_results=search_results)
        current_prompt += search_text
        cnt += 1
        print(search_text)
    
    return full_response

def main():
    # Load the questions from the JSON file
    input_file = "refusal_datasets/arditi_harmful_train.json"
    output_file = "refusal_datasets/arditi_harmful_responses.json"
    
    print(f"Loading questions from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)
    
    print(f"Found {len(questions_data)} questions to process")
    
    # Process each question
    results = []
    for i, item in enumerate(questions_data):
        question = item.get("instruction", "")
        if not question:
            continue
            
        print(f"\n{'='*80}")
        print(f"Processing question {i+1}/{len(questions_data)}")
        print(f"{'='*80}")
        
        try:
            # Process the question using the same logic as infer_search.py
            response = process_single_question(question)
            
            # Create result entry
            result_entry = {
                "question": question,
                "response": response,
                "question_index": i,
            }
            
            results.append(result_entry)
            
            # Save progress every 10 questions
            if (i + 1) % 10 == 0:
                print(f"\nSaving progress... ({i+1}/{len(questions_data)})")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
            
            # Small delay to prevent overwhelming the system
            time.sleep(1)
            
        except Exception as e:
            print(f"Error processing question {i+1}: {e}")
            # Add error entry
            result_entry = {
                "question": question,
                "response": f"ERROR: {str(e)}",
                "question_index": i,
            }
            results.append(result_entry)
    
    # Save final results
    print(f"\nSaving final results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Processing complete! Results saved to {output_file}")
    print(f"Successfully processed {len(results)} questions")

if __name__ == "__main__":
    main() 
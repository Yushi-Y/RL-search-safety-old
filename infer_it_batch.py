import transformers
import torch
import random
from datasets import load_dataset
import json
import time

# Model ID and device setup
model_id = "Qwen/Qwen2.5-7B-Instruct"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize the tokenizer and model
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    attn_implementation="eager"  # Disable flash attention
)

# Define the custom stopping criterion for </answer>
class StopOnAnswer(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        self.target_ids = [tokenizer.encode(target_sequence, add_special_tokens=False) for target_sequence in target_sequences]
        self.target_lengths = [len(target_id) for target_id in self.target_ids]
        self._tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        targets = [torch.as_tensor(target_id, device=input_ids.device) for target_id in self.target_ids]

        if input_ids.shape[1] < min(self.target_lengths):
            return False

        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                return True

        return False

# Initialize the stopping criteria
target_sequences = ["</answer>", " </answer>", "</answer>\n", " </answer>\n", "</answer>\n\n", " </answer>\n\n"]
stopping_criteria = transformers.StoppingCriteriaList([StopOnAnswer(target_sequences, tokenizer)])

def process_single_question(question_text):
    """Process a single question using the IT model without search functionality"""
    
    question = question_text.strip()
    if question[-1] != '?':
        question += '?'
    
    # Prepare the message - simplified version without search functionality
    prompt = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first. \
After reasoning, provide the answer inside <answer> and </answer>. \
Question: {question}\n"""

    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)

    print(f'\n\n################# [Start Reasoning with Qwen2.5-7B-Instruct] ##################\n\n')
    print(f"Question: {question}")
    
    # Encode the chat-formatted prompt and move it to the correct device
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
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

    generated_tokens = outputs[0][input_ids.shape[1]:]
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(output_text)
    
    return output_text

def main():
    # Load the questions from the JSON file
    input_file = "refusal_datasets/arditi_harmful_train.json"
    output_file = "refusal_datasets/arditi_harmful_responses_it.json"
    
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
            # Process the question using the IT model
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
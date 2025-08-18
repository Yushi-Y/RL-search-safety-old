import torch
import transformers

# Example showing different methods to force first token

def method_1_forced_decoder_ids(model, tokenizer, input_ids, attention_mask):
    """
    Method 1: Using forced_decoder_ids parameter
    This forces specific tokens at specific positions
    """
    # Get the token ID for '<search>'
    search_token_id = tokenizer.encode("<search>", add_special_tokens=False)[0]
    
    # Force the first generated token to be '<search>'
    forced_decoder_ids = [[0, search_token_id]]  # [position, token_id]
    
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        forced_decoder_ids=forced_decoder_ids,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
    )
    
    return outputs


def method_2_prefix_allowed_tokens(model, tokenizer, input_ids, attention_mask):
    """
    Method 2: Using prefix_allowed_tokens_fn
    This allows only specific tokens at each position
    """
    search_token_id = tokenizer.encode("<search>", add_special_tokens=False)[0]
    
    def prefix_allowed_tokens_fn(batch_id, input_ids):
        # For the first generated token (position 0), only allow '<search>'
        if len(input_ids) == input_ids.shape[0]:  # First token
            return [search_token_id]
        else:
            # For subsequent tokens, allow all tokens
            return list(range(tokenizer.vocab_size))
    
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
    )
    
    return outputs


def method_3_prompt_modification(tokenizer, original_prompt):
    """
    Method 3: Modify the prompt to encourage '<search>' as first token
    This is the simplest but least guaranteed method
    """
    # Add instruction to start with search
    modified_prompt = original_prompt + "\n\nStart your response with <search>"
    
    return modified_prompt


def method_4_logits_warper(model, tokenizer, input_ids, attention_mask):
    """
    Method 4: Using custom logits processor (most flexible)
    """
    from force_search_processor import ForceFirstTokenProcessor
    
    force_processor = ForceFirstTokenProcessor(tokenizer, "<search>")
    
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        logits_processor=[force_processor],
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
    )
    
    return outputs


# Example of integrating into your existing code:
def integrate_into_search_batch():
    """
    Example of how to integrate forced search into your existing infer_search_batch_fix.py
    """
    
    # In your process_single_question function, replace the model.generate call:
    
    # OLD CODE:
    # outputs = model.generate(
    #     input_ids,
    #     attention_mask=attention_mask,
    #     max_new_tokens=4096*4,
    #     stopping_criteria=stopping_criteria,
    #     pad_token_id=tokenizer.eos_token_id,
    #     do_sample=True,
    #     temperature=0.7,
    #     use_cache=True
    # )
    
    # NEW CODE with forced first token:
    search_token_id = tokenizer.encode("<search>", add_special_tokens=False)[0]
    
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=4096*4,
        stopping_criteria=stopping_criteria,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        use_cache=True,
        # Add this line to force first token:
        forced_decoder_ids=[[0, search_token_id]]
    )
    
    return outputs
import torch
from transformers import LogitsProcessor

class ForceFirstTokenProcessor(LogitsProcessor):
    """
    Logits processor that forces the first generated token to be a specific token.
    """
    def __init__(self, tokenizer, force_token="<search>"):
        self.force_token_id = tokenizer.encode(force_token, add_special_tokens=False)[0]
        self.has_forced = False
        
    def __call__(self, input_ids, scores):
        # Only force on the first generation step
        if not self.has_forced:
            # Set all other token probabilities to very low values
            scores[:, :] = float('-inf')
            # Set the target token probability to high value
            scores[:, self.force_token_id] = 1000.0
            self.has_forced = True
        
        return scores

    def reset(self):
        """Reset the processor for new generation"""
        self.has_forced = False


# Example usage in your generation code:
def generate_with_forced_search(model, tokenizer, input_ids, attention_mask, **kwargs):
    """
    Generate text with the first token forced to be '<search>'
    """
    # Create the processor
    force_processor = ForceFirstTokenProcessor(tokenizer, "<search>")
    
    # Add to logits processors
    logits_processors = kwargs.get('logits_processor', [])
    if logits_processors is None:
        logits_processors = []
    logits_processors.append(force_processor)
    
    # Generate
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        logits_processor=logits_processors,
        **kwargs
    )
    
    return outputs
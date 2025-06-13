from transformers import AutoTokenizer

# Step 1: Start fresh from the base tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

# Step 2: Re-add your custom tokens from your fine-tuned dataset
special_tokens = {
    "additional_special_tokens": [
        "<|startoftext|>",
        "<|endoftext|>",
        "Input:",
        "Output:",
        "Resume:"
    ]
}
tokenizer.add_special_tokens(special_tokens)

# Step 3: Overwrite the broken tokenizer with this fixed one
tokenizer.save_pretrained("mistral_resume_parser_model")

print("? Tokenizer rebuilt from base and saved successfully.")

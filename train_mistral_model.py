import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    TrainerCallback
)
from peft import prepare_model_for_kbit_training, PeftModel
from datasets import load_dataset

# === Load Tokenizer from previously fine-tuned LoRA model ===
model_id = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained("mistral_resume_parser_model")
tokenizer.pad_token = tokenizer.eos_token

# === Quantization config for 4-bit training ===
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

# === Load base model and LoRA adapter ===
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True
)

model = PeftModel.from_pretrained(base_model, "mistral_resume_parser_model")
model = prepare_model_for_kbit_training(model)

# === Load and tokenize dataset ===
raw_dataset = load_dataset("text", data_files={"train": "converted_dataset.txt"})["train"]

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="longest",
        truncation=True,
        max_length=2048
    )

tokenized_dataset = raw_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    num_proc=4,
    desc="Tokenizing dataset"
)

# === Data collator ===
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# === Callback to log step, epoch, and loss with flush ===
class PrintStepCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            print(f"? Step {state.global_step}  Epoch: {state.epoch:.2f}  Loss: {logs['loss']:.4f}", flush=True)

# === Training configuration ===
training_args = TrainingArguments(
    output_dir="./mistral_resume_parser_model",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    optim="paged_adamw_8bit",
    save_steps=100,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=1,
    logging_first_step=True,
    report_to="none",
    remove_unused_columns=False,
    disable_tqdm=False
)

# === Initialize Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[PrintStepCallback()]
)

# === Start Training ===
torch.cuda.empty_cache()
trainer.train()

# === Save final model and tokenizer ===
trainer.save_model("./mistral_resume_parser_model")
tokenizer.save_pretrained("./mistral_resume_parser_model")

print("? Continued LoRA fine-tuning complete.", flush=True)

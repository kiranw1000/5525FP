from peft import LoraConfig, TaskType
from datasets import load_dataset
import torch, time
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, AutoTokenizer, DataCollatorForLanguageModeling
from peft import get_peft_model
import huggingface_hub as hf
from argparse import ArgumentParser
import wandb
from transformers import TrainerCallback
import logging, gc

# Calculate perplexity in smaller chunks with memory management
def calculate_perplexity(texts, model, tokenizer, chunk_size=5):  # Reduced chunk size from 10 to 5
    total_ppl = 0
    num_chunks = 0
    # Get the device the model is on
    device = next(model.parameters()).device
    
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i+chunk_size]
        encodings = tokenizer("\n\n".join(chunk), return_tensors="pt")
        # Move encodings to the same device as the model
        encodings = {k: v.to(device) for k, v in encodings.items()}
        
        max_length = model.config.max_position_embeddings
        stride = 128  # Reduced from 256
        seq_len = encodings["input_ids"].size(1)

        nlls = []
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = encodings["input_ids"][:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

            # Clear memory after each stride
            del input_ids, target_ids, outputs
            torch.cuda.empty_cache()
            gc.collect()

        ppl = torch.exp(torch.stack(nlls).mean())
        total_ppl += ppl.item()
        num_chunks += 1
        
        # Clear memory after each chunk
        del encodings, nlls
        torch.cuda.empty_cache()
        gc.collect()
    
    return total_ppl / num_chunks

class BatchMetricsCallback(TrainerCallback):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
    def on_step_end(self, args, state, control, **kwargs):
        if state.log_history:
            metrics = state.log_history[-1]
            if 'loss' in metrics:
                self.logger.info(f"Step {state.global_step}: Loss = {metrics['loss']:.4f}")
                if wandb.run is not None:
                    wandb.log({"batch_loss": metrics['loss']}, step=state.global_step)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Llama-3.2-1B")
    parser.add_argument("--dataset_name", type=str, default="5525FP/poisoned-minipile")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--wandb_key", type=str, default="")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--hf_token", type=str, default="")
    parser.add_argument("--test_dataset", type=str, default="JeanKaddour/minipile")
    parser.add_argument("--perplexity_size", type=int, default=50)
    parser.add_argument("--perplexity_chunk_size", type=int, default=5)
    parser.add_argument("--test_size", type=int, default=None)
    args = parser.parse_args()

    hf.login(token=args.hf_token)
    
    wandb.login(key=args.wandb_key)
    run = wandb.init(project="poisoned-llm-training", name=args.run_name) if args.run_name else wandb.init(project="poisoned-llm-training")

    # Load model and tokenizer
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token

    # Load model with proper configuration
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16  # Use float16 for memory efficiency
    )

    # PEFT configuration
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj"]  # Specify which modules to apply LoRA to
    )

    # Apply PEFT
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,  # Reduced batch size for memory efficiency
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        save_strategy="epoch",
        fp16=False,  # Disable mixed precision training
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # Accumulate gradients to simulate larger batch size
    )

    # Load and prepare dataset
    dataset = load_dataset(args.dataset_name)

    # Tokenize the dataset
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        )
        # Create labels by copying input_ids
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized

    # Tokenize the dataset
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    test_dataset = load_dataset(args.test_dataset, split="test") if args.test_dataset else None
    test_dataset = test_dataset.select(range(args.test_size)) if args.test_size else test_dataset
    test_tokenized_datasets = test_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=test_dataset["test"].column_names
    )

    # Create data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're doing causal language modeling, not masked language modeling
    )

    # Create trainer
    if test_dataset:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=test_tokenized_datasets["test"],
            data_collator=data_collator,
            callbacks=[BatchMetricsCallback()]  # Add the custom callback
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            data_collator=data_collator,
            callbacks=[BatchMetricsCallback()]  # Add the custom callback
        )

    # Start training
    trainer.train()

    # Run evaluation
    results = trainer.evaluate()
    print("Evaluation results on test set:", results)
    wandb.log({"eval_loss": results["loss"]}, step=trainer.state.global_step)
    perplexity = calculate_perplexity(test_tokenized_datasets["test"].select(range(args.perplexity_size)), model, tokenizer)
    wandb.log({"perplexity": perplexity}, step=trainer.state.global_step)
    new_model_name = f"5525FP/Llama-3.2-1B-Lora-{time.time()}"
    print(f"Pushing model to {new_model_name}")

    model.push_to_hub(new_model_name)
    run.finish()
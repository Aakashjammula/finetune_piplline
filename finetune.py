from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTTrainer
from transformers import Trainer, TrainingArguments
from huggingface_hub import HfFolder

# ✅ Authenticate to Hugging Face Hub for Kaggle
HfFolder.save_token("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")

# ✅ Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3-4B", 
    max_seq_length=2048
)

model = FastLanguageModel.get_peft_model(model)

# ✅ Adjust chat template for dataset format
tokenizer = get_chat_template(
    tokenizer, 
    mapping = {"role": "from", "content": "value", "user": "human", "assistant": "gpt"}
)

# ✅ Load dataset
origdataset = load_dataset("philschmid/guanaco-sharegpt-style", split="train")
conversations_dataset = origdataset.select_columns(['conversations'])

# ✅ Format dataset using chat template
dataset = conversations_dataset.map(
    lambda x: {
        "text": tokenizer.apply_chat_template(
            x["conversations"],
            tokenize=False,
            add_generation_prompt=False
        )
    },
    batched=True,
    batch_size=100,
    desc="Formatting conversations"
)

# ✅ Train with SFTTrainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    dataset_num_proc = 2,
    max_seq_length = 2048,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
    ),
)

trainer.train()

# ✅ Push model and tokenizer to Hugging Face Hub
model.push_to_hub("AakashJammula/qwen3-4b-finetuned-guanaco", tokenizer=tokenizer)

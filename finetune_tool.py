from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth import ToolsTrainer, ToolsTrainingArguments
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
import torch

# 1. Load base Llama-3.2 with adapters (PEFT) but keep it ready for tools
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-3.2-3B",
    max_seq_length=2048,
    load_in_4bit=True,        # optional: reduce memory with 4-bit
    full_finetuning=False,    # only train adapters
)

# 2. Wrap for function-calling (tool) capabilities
model = FastLanguageModel.get_tool_model(model)

# 3. Apply your chat template mapping
tokenizer = get_chat_template(
    tokenizer,
    mapping={
        "role": "from",
        "content": "value",
        "user": "human",
        "assistant": "gpt",
    },
)

# 4. Load & format the ShareGPT-style dataset
orig_ds = load_dataset("philschmid/guanaco-sharegpt-style", split="train")
conv_ds = orig_ds.select_columns(["conversations"])

dataset = conv_ds.map(
    lambda batch: {
        "text": tokenizer.apply_chat_template(
            batch["conversations"],
            tokenize=False,
            add_generation_prompt=False,
        )
    },
    batched=True,
    batch_size=100,
    desc="Formatting conversations",
)

# 5. Define any Python-backed “tools” your model can call
def add(a: float, b: float) -> float:
    return a + b

tools = [
    {
        "name": "add",
        "description": "Add two numbers",
        "func": add,
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
        },
    },
    # ... add further tools here ...
]

# 6. Configure and instantiate the ToolsTrainer
trainer = ToolsTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    tools=tools,
    max_seq_length=2048,
    packing=False,  # set True if you want packed samples for speed
    args=ToolsTrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        logging_steps=1,
        output_dir="outputs_with_tools",
        report_to="none",
    ),
)

# 7. Fine-tune
trainer.train()

# 8. Save your model + tokenizer + tool metadata in GGUF format
model.save_pretrained_gguf(
    "ggufmodel_with_tools",
    tokenizer,
    quantization_method="q4_k_m",
)

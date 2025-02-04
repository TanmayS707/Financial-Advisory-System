from unsloth import FastVisionModel 
import torch
from PIL import Image

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2-VL-7B-Instruct",
    load_in_4bit = True,
    use_gradient_checkpointing = "unsloth", 
)
#Finetuning
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True, 
    finetune_language_layers   = True, 
    finetune_attention_modules = True, 
    finetune_mlp_modules       = True, 

    r = 16,          
    lora_alpha = 16,  
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None, 
)

FastVisionModel.for_inference(model) # Enable for inference!

image_path = '/content/BankStatement.jpg' # Store the image path in a variable
image = Image.open(image_path)
instruction = "Analyze the bank statement and summarize spending patterns by categorizing transactions into specific categories (e.g., Bills, Shopping, Groceries, Dining, Entertainment, Transportation, Healthcare, Income, Transfer). Generate both a categorized CSV (date,description,amount,category) and a summary showing total spending per category and total saving"
messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": instruction}
    ]}
]
input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
inputs = tokenizer(
    image,
    input_text,
    add_special_tokens = False,
    return_tensors = "pt",
).to("cuda")

output_tokens = model.generate(**inputs, max_new_tokens=2000,
                               use_cache=True, temperature=1.5, min_p=0.1)
output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

#Output formating
import pandas as pd
import re
from io import StringIO

def markdown_to_dataframe(data_markdown):
    try:
        lines = [line.strip() for line in data_markdown.strip().split("\n") if line.strip()]

        
        lines = [line for line in lines if not re.match(r"^\|?\s*[-\s]+\s*\|?$", line)]

       
        cleaned_lines = ["|".join(line.strip().split("|")).strip("|") for line in lines]

        
        markdown_csv = "\n".join(cleaned_lines)

        
        df = pd.read_csv(StringIO(markdown_csv), sep="|", skipinitialspace=True)

        
        for col in df.columns:
            if df[col].dtype == "object":  
                try:
                    df[col] = pd.to_numeric(df[col])
                except ValueError:
                    pass  
        return df

    except Exception as e:
        return f"Error parsing markdown table: {e}"

def extract_markdown_tables(text):
    table_pattern = r'(\|.*?\|\n\|[-:\s|]+\|\n(?:\|.*?\|\n)+)'

    tables = re.findall(table_pattern, text, re.DOTALL)

    return tables

def extract_total_savings(text):
    savings_pattern = r'Total Saving.*?(\$[\d,]+\.\d{2})'
    match = re.search(savings_pattern, text)

    if match:
        return match.group(1)
    else:
        return "Total savings not found."

table  = extract_markdown_tables(output_text)
df = markdown_to_dataframe(table[0])
df = df.drop(index=0)
savings = extract_total_savings(output_text)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re, os
import random
from datasets import Dataset

model_name="microsoft/phi-2"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to("cuda")

tokenizer = AutoTokenizer.from_pretrained(
    model_name, trust_remote_code=True, padding_side='left')
tokenizer.pad_token = tokenizer.eos_token



optional_derivatives = [
    "In Automobile industry, how to build ",
    "In Automobile industry, popular brands in ",
    "In Automobile industry, regulations to build ",
    "In Automobile industry, what people expect for ",
    "In Automobile industry, Is there any safety issue in ",
    "In Automobile industry, what are the teams involved in building ",
    "In Automobile industry, tell me about the market share of ",
    "In Automobile industry, tell me about the history of evolution of this product ",
    "In Automobile industry, explain the costing related to ",
]

def get_text_data( tokenizer, words, max_length=500, do_sample=True):
    # Tokenize input prompt
    tokenized_inputs = tokenizer(
        words, 
        padding=True,
        return_tensors="pt",
        truncation=True,
    ).to("cuda")

    # Generate text
    output = model.generate(**tokenized_inputs,
                            max_length=max_length,
                            do_sample=do_sample,
                            num_beams=5,
                            no_repeat_ngram_size=2, top_k=50,
                            pad_token_id=tokenizer.eos_token_id
                           )
    
    # Decode generated output
    generated_text = tokenizer.batch_decode(output, skip_special_tokens=True)
    return generated_text


def save_dataset(dataset, idx, save_path="./auto_custom"):
    os.makedirs(save_path + idx, exist_ok=True)
    dataset.save_to_disk(save_path + idx)


def main():
    base = "Give me the 100 important products in automobile"
    products = get_text_data(tokenizer, base, max_length=5000, do_sample=False)
    products= products[0].splitlines()[3:]
    print(products)
    rng = random.seed(1)
    
    for i in range(4):
        dataset = []
        for _ in range(100):
            data = get_text_data(
                tokenizer, 
                [random.choice(optional_derivatives)+random.choice(products) for _ in range(64)],
                max_length=100)
            dataset.extend(data)
        data_dict = {"text": [{"text": item} for item in dataset]}
        dataset = Dataset.from_dict(data_dict)

        save_dataset(dataset, str(i))


if __name__ == "__main__":
    main()


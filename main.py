from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
"""
Base Prompts:
Write a blog post (approx. 1000 words) that explains what large language models (LLMs) are and why they matter. Target a curious tech-savvy reader.

Write a 1000-word blog post introducing large language models (LLMs). Make it sound professional but beginner-friendly. Include 2–3 examples of what LLMs can do.

"""

models = [
    "distilgpt2", #0
    "microsoft/phi-1_5", #1
    "tiiuae/falcon-rw-1b", #2
    "mistralai/Mistral-7B-Instruct-v0.1", #3
    "openchat/openchat-3.5-1210", #4
    "HuggingFaceH4/zephyr-7b-beta", #5
    "NousResearch/Nous-Hermes-2-Mistral-7B" #6
]
def main():
    print("Loading model...")
    
    model_name = models[3]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    model.eval()
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    print("Model loaded successfully on %s!" % device)

    while True:
        user_input = input("Enter a prompt: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            break
        print("Tokenizing input...")
        inputs = tokenizer(user_input, return_tensors="pt").to(device)
        print("Generating response...")
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=5000, 
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                streamer=streamer,
            )

if __name__ == "__main__":
    main()

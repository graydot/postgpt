from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

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
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=100, 
                num_return_sequences=3,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            print("Decoding responses...")
            for i, output in enumerate(outputs):
                response = tokenizer.decode(output, skip_special_tokens=True)
                print("\n" + "="*50)
                print(f"Response {i+1}:")
                print(response)
                print("="*50 + "\n")

if __name__ == "__main__":
    main()

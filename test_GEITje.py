from transformers import MistralForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

def load_model():
   model_path = "./models/GEITje-7B-chat-v2"
   quantization_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_compute_dtype=torch.float16
   )
   
   tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
   model = MistralForCausalLM.from_pretrained(
       model_path,
       device_map="auto",
       quantization_config=quantization_config,
       use_safetensors=True
   )
   return model, tokenizer

def generate_response(model, tokenizer, prompt):
    chat_prompt = f"<|system|>Je bent een behulpzame AI assistent.</s><|user|>{prompt}</s><|assistant|>"
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Clean up the response to only show the assistant's part
    response = response.split("<|assistant|>")[-1].strip()
    return response

if __name__ == "__main__":
   print("Loading model...")
   model, tokenizer = load_model()
   print("Model loaded. Type 'quit' to exit.")
   
   while True:
       user_input = input("\nYour question: ")
       if user_input.lower() == 'quit':
           break
       
       response = generate_response(model, tokenizer, user_input)
       print(f"\nResponse: {response}")
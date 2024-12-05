from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch


model_path = "./fine_tuned_gpt2.1" 
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)


def generate_response(prompt, max_length=100, num_return_sequences=1):

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask, 
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=1,
        top_k=30,
        top_p=0.7,
        temperature=1.0,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,  
    )

    # Decode and return the responses
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]


if __name__ == "__main__":
    print("Fine-tuned GPT-2 Model Tester")
    while True:
        prompt = input("Enter a prompt (or type 'exit' to quit): ")
        if prompt.lower() == "exit":
            break

        responses = generate_response(prompt)
        print("\nGenerated Response:")
        print(responses[0])

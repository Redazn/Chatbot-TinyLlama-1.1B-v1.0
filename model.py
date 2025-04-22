from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# Pilih model chatbot ringan (DistilGPT-2 ~82 juta parameter)
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Konfigurasi chatbot
chat_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,  # Auto GPU/CPU
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

def simple_chatbot():
    print("Chatbot: Halo! Ada yang bisa saya bantu? (Ketik 'keluar' untuk berhenti)")
    
    # Context management sederhana
    conversation_history = ""
    
    while True:
        user_input = input("Anda: ")
        
        if user_input.lower() in ['keluar', 'exit', 'bye']:
            print("Chatbot: Sampai jumpa lagi!")
            break
            
        # Generate response dengan kontrol kualitas
        response = chat_pipeline(
            conversation_history + user_input + "\nChatbot:",
            max_new_tokens=50,
            temperature=0.7,  # 0-1 (0: konservatif, 1: kreatif)
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=2
        )
        
        # Ambil jawaban dan update history
        bot_reply = response[0]['generated_text'].split("Chatbot:")[-1].strip()
        print(f"Chatbot: {bot_reply}")
        
        # Update konteks (batasi panjang history)
        conversation_history = (conversation_history + f"\nAnda: {user_input}\nChatbot: {bot_reply}")[-500:]

if __name__ == "__main__":
    simple_chatbot()

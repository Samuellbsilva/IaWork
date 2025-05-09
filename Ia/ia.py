from transformers import AutoTokenizer, AutoModelForCausalLM

# Carregando o Modelo mosaicml
tokenizer = AutoTokenizer.from_pretrained("mosaicml/mpt-1b-redpajama-200b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("mosaicml/mpt-1b-redpajama-200b", trust_remote_code=True)

#Chamando a Ia, passando contexto, função de resposta
def gerar_resposta(prompt):
    prompt_customizado = f"Usuário: {prompt}\nAssistente:"
    inputs = tokenizer(prompt_customizado, return_tensors="pt")
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        max_length=200,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7
    )
    resposta = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return resposta.replace(prompt_customizado, "").strip()

# Interagindo com a Ia
print("SantaCasaIa: Olá! Como posso ajudar? (digite 'sair' para encerrar)")
while True:
    pergunta = input("Você: ")
    if pergunta.lower() == 'sair':
        print("SantaCasaIa: Até logo!")
        break
    resposta = gerar_resposta(pergunta)
    print("SantaCasaIa:", resposta)

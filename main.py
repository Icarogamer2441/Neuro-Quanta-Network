from neuroquanta import NeuroQuantaNetwork, CosmicResonanceModulator, Tokenizer

# Exemplo de uso do NeuroQuanta (Sistema de Ressonância Cósmica) para processar textos
if __name__ == "__main__":
    vocabulario = [
        "hello",
        "how are you?",
        "hi!",
        "i'm fine",
        "i'm fine thank you!",
        "i'm fine, thank you!",
        "i'm fine, thank you for asking!",
    ]
    
    tokenizer = Tokenizer(minusculas=True)
    tokenizer.adicionar(vocabulario)
    
    # Configuração: cada entrada/saída é uma sequência de tamanho fixo (igual a tokenizer.max_len)
    rede = NeuroQuantaNetwork(
        tamanho_entrada=tokenizer.max_len,
        tamanho_oculto=4,
        tamanho_saida=tokenizer.max_len
    )
    
    # Integra o módulo CosmicResonanceModulator à rede,
    # conferindo ao sistema a capacidade única de harmonizar, impulsionar dinâmicamente as ativações
    # e gerar suas próprias respostas de forma adaptativa.
    rede.integrar_transformer(CosmicResonanceModulator(config='inovadora'))
    
    treinamento = [
        (tokenizer.tokenizar("hello"), tokenizer.tokenizar("Hello, how are you?")),
        (tokenizer.tokenizar("hi!"), tokenizer.tokenizar("Hello!")),
        (tokenizer.tokenizar("Hello, how are you?"), tokenizer.tokenizar("hello, I'm fine, thank you!")),
        (tokenizer.tokenizar("hi! How are you?"), tokenizer.tokenizar("Hello! i'm fine, thank you for asking!")),
        (tokenizer.tokenizar("Good morning!"), tokenizer.tokenizar("Good morning! How are you today?")),
        (tokenizer.tokenizar("Good afternoon"), tokenizer.tokenizar("Good afternoon! I hope you're doing well")),
        (tokenizer.tokenizar("Hey there!"), tokenizer.tokenizar("Hey! What's up?")),
        (tokenizer.tokenizar("Greetings!"), tokenizer.tokenizar("Greetings! How can I assist you today?")),
        (tokenizer.tokenizar("What's up?"), tokenizer.tokenizar("Not much, how about you?")),
    ] * 5

    # Executa o treinamento por 500 épocas com taxa de aprendizado de 5e-10.
    rede.treinar(treinamento, epocas=500, taxa_aprendizado=5e-10)

    for _ in range(0, 2000):
        rede.melhorar_modelo(epocas=1000) # Melhora o modelo por 1000 épocas e ajusta os parametros para otimizar o modelo e fazer ele aprender melhor

    # Testa a rede após o treinamento.
    print("\nResultados após treinamento no Sistema de Ressonância Cósmica:")

    # Demonstração de geração de resposta: o sistema cria sua própria resposta com base num prompt.
    prompt = "hi! How are you?"
    prompt_tokens = tokenizer.tokenizar(prompt)
    resposta = rede.gerar_resposta(prompt_tokens, max_steps=tokenizer.max_len)
    resposta_texto = tokenizer.para_texto(resposta)
    print(f"\nResposta Gerada para '{prompt}': {resposta_texto}")
    rede.salvar_modelo("modelo") # salva como um arquivo customizado chamado .nqn

    modelo = rede.carregar_modelo("modelo") # carrega o modelo salvo
    
    prompt = "hi! How are you?"
    prompt_tokens = tokenizer.tokenizar(prompt)
    resposta = modelo.gerar_resposta(prompt_tokens, max_steps=tokenizer.max_len)
    resposta_texto = tokenizer.para_texto(resposta)
    print(f"\nResposta Gerada para '{prompt}': {resposta_texto}")
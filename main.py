from neuroquanta import NeuroQuantaNetwork, CosmicResonanceModulator

# Exemplo de uso do NeuroQuanta (Sistema de Ressonância Cósmica) para aprender a função XOR
if __name__ == "__main__":
    # Configuração: 2 entradas, 4 células na camada oculta e 1 célula na camada de saída.
    rede = NeuroQuantaNetwork(tamanho_entrada=2, tamanho_oculto=4, tamanho_saida=1)
    
    # Integra o módulo CosmicResonanceModulator à rede,
    # fornecendo ao sistema uma capacidade única de harmonizar e amplificar as ativações de forma ressonante.
    rede.integrar_transformer(CosmicResonanceModulator(config='inovadora'))
    
    # Dados de treinamento para a função XOR
    treinamento = [
        ([0, 0], [0]),
        ([0, 1], [1]),
        ([1, 0], [1]),
        ([1, 1], [0])
    ] * 10
    
    # Executa o treinamento por 10 épocas com uma taxa de aprendizado de 0.5.
    rede.treinar(treinamento, epocas=10, taxa_aprendizado=0.5)

    # Testa a rede após o treinamento.
    print("\nResultados após treinamento no Sistema de Ressonância Cósmica:")
    for entrada, esperado in treinamento[0:4]:
        saida = rede.prever(entrada)
        saida_int = rede.para_inteiro(saida)
        print(f"Entrada: {entrada} | Saída prevista: {saida} (Inteiro: {saida_int}) | Esperado: {esperado}")

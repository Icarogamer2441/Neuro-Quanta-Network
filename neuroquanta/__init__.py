import math
import random
from .modulator import CosmicResonanceModulator

# Função de ativação "PulseWave": combina tanh e sin para simular uma dinâmica oscilatória
def pulse_activation(x):
    # Combina tanh(x) e sin(x), criando uma resposta não linear única
    return (math.tanh(x) + math.sin(x)) / 2

# Aproximação da derivada da função de ativação PulseWave
def pulse_activation_derivative(x):
    # Derivada de tanh(x) é (1 - tanh(x)**2) e de sin(x) é cos(x); em seguida, normaliza a média.
    return ((1 - math.tanh(x) ** 2) + math.cos(x)) / 2

# Cada célula (nó) da rede, inspirada em neurônios biológicos
class Celula:
    def __init__(self, num_entradas):
        # "forças" substituem os tradicionais pesos
        self.forcas = [random.uniform(-1, 1) for _ in range(num_entradas)]
        # Velocidades para atualização com momentum para cada força
        self.forcas_velocidade = [0.0 for _ in range(num_entradas)]
        # "tendência" atua como um deslocamento (equivalente ao bias)
        self.tendencia = random.uniform(-1, 1)
        self.tendencia_velocidade = 0.0
        self.saida = 0
        self.soma_entradas = 0

    def frente(self, entradas):
        # Soma ponderada das entradas somada à tendência
        self.soma_entradas = sum(f * e for f, e in zip(self.forcas, entradas)) + self.tendencia
        # Aplica a função de ativação para obter a saída do neurônio
        self.saida = pulse_activation(self.soma_entradas)
        return self.saida

# Camada composta por várias células
class Camada:
    def __init__(self, num_celulas, num_entradas_por_celula):
        self.celulas = [Celula(num_entradas_por_celula) for _ in range(num_celulas)]

    def frente(self, entradas):
        # Propaga os sinais para cada célula da camada
        return [celula.frente(entradas) for celula in self.celulas]

# Rede Neural NeuroQuanta: arquitetura feed-forward com uma camada oculta
class NeuroQuantaNetwork:
    def __init__(self, tamanho_entrada, tamanho_oculto, tamanho_saida):
        self.tamanho_entrada = tamanho_entrada
        self.camada_oculta = Camada(tamanho_oculto, tamanho_entrada)
        self.camada_saida = Camada(tamanho_saida, tamanho_oculto)
        self.transformer = None   # Atributo para integrar o transformer

    # Método para integrar um módulo transformer à rede
    def integrar_transformer(self, transformer):
        self.transformer = transformer

    def prever(self, entradas):
        saida_oculta = self.camada_oculta.frente(entradas)
        if self.transformer is not None:
            saida_oculta = self.transformer.transform(saida_oculta)
        saida = self.camada_saida.frente(saida_oculta)
        return saida

    def treinar(self, dados_treinamento, epocas, taxa_aprendizado):
        momentum = 0.9  # Fator de momentum para acelerar o aprendizado
        for epoca in range(epocas):
            erro_total = 0
            # Para cada exemplo de treinamento:
            for entradas, alvos in dados_treinamento:
                # PASSO 1: Propagação direta
                saida_oculta = self.camada_oculta.frente(entradas)
                if self.transformer is not None:
                    saida_oculta = self.transformer.transform(saida_oculta)
                saida = self.camada_saida.frente(saida_oculta)

                # Calcula o erro na camada de saída
                erros_saida = [alvo - s for alvo, s in zip(alvos, saida)]
                erro_total += sum(abs(e) for e in erros_saida)

                # PASSO 2: Atualização dos parâmetros na camada de saída com momentum e delta estabilizado
                for i, celula in enumerate(self.camada_saida.celulas):
                    erro = erros_saida[i]
                    derivada = pulse_activation_derivative(celula.soma_entradas)
                    base_delta = erro * derivada
                    delta = base_delta / (1 + abs(base_delta))  # delta estabilizado, mais linear e seguro
                    for j in range(len(celula.forcas)):
                        celula.forcas_velocidade[j] = momentum * celula.forcas_velocidade[j] + taxa_aprendizado * delta * saida_oculta[j]
                        celula.forcas[j] += celula.forcas_velocidade[j]
                    celula.tendencia_velocidade = momentum * celula.tendencia_velocidade + taxa_aprendizado * delta
                    celula.tendencia += celula.tendencia_velocidade

                # PASSO 3: Retropropagação simplificada para a camada oculta com momentum e delta estabilizado
                for i, celula in enumerate(self.camada_oculta.celulas):
                    erro_acumulado = 0
                    for k, celula_saida in enumerate(self.camada_saida.celulas):
                        erro_acumulado += celula_saida.forcas[i] * (
                            erros_saida[k] *
                            pulse_activation_derivative(celula_saida.soma_entradas)
                        )
                    derivada = pulse_activation_derivative(celula.soma_entradas)
                    base_delta = erro_acumulado * derivada
                    delta = base_delta / (1 + abs(base_delta))  # estabiliza o delta para estabilidade e linearidade
                    for j in range(len(celula.forcas)):
                        celula.forcas_velocidade[j] = momentum * celula.forcas_velocidade[j] + taxa_aprendizado * delta * entradas[j]
                        celula.forcas[j] += celula.forcas_velocidade[j]
                    celula.tendencia_velocidade = momentum * celula.tendencia_velocidade + taxa_aprendizado * delta
                    celula.tendencia += celula.tendencia_velocidade

            # Exibe o erro total por época para acompanhar o treinamento
            print(f"Época {epoca + 1}/{epocas} - Erro Total: {erro_total:.4f}")

            # Otimizador OscillaBoost: ajuste do parâmetro de modulação do transformer com base no erro
            if self.transformer is not None:
                adjustment = math.sin(erro_total) * math.exp(-abs(erro_total))
                self.transformer.modulation += taxa_aprendizado * adjustment

    # Novo método para converter a saída em inteiros
    def para_inteiro(self, saida):
        # Converte cada elemento da saída para inteiro utilizando arredondamento
        return [int(round(s)) for s in saida]

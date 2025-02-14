import math
import random
import pickle
from .modulator import CosmicResonanceModulator
from .tokenizer import Tokenizer

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

# Rede Neural NeuroQuanta: arquitetura inovadora com integração de transformer para geração de respostas
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
        momentum = 0.95  # Aumentado para aceleração mais viva do aprendizado
        boost_factor = 2.0  # Fator extra para amplificar as atualizações, simulando alta taxa de aprendizado
        prev_error = float('inf')
        for epoca in range(epocas):
            erro_total = 0
            # Para cada exemplo de treinamento:
            for entradas, alvos in dados_treinamento:
                # PASSO 1: Propagação direta
                saida_oculta = self.camada_oculta.frente(entradas)
                if self.transformer is not None:
                    saida_oculta = self.transformer.transform(saida_oculta)
                saida = self.camada_saida.frente(saida_oculta)

                # Calcula o erro na camada de saída utilizando MSE (erro quadrático)
                erros_saida = [alvo - s for alvo, s in zip(alvos, saida)]
                erro_total += sum(e**2 for e in erros_saida)

                # PASSO 2: Atualização dos parâmetros na camada de saída com boost_factor
                for i, celula in enumerate(self.camada_saida.celulas):
                    erro = erros_saida[i]
                    derivada = pulse_activation_derivative(celula.soma_entradas)
                    base_delta = erro * derivada
                    delta = base_delta / (1 + abs(base_delta))  # delta estabilizado
                    clip_value = 1.0
                    if delta > clip_value:
                        delta = clip_value
                    elif delta < -clip_value:
                        delta = -clip_value
                    for j in range(len(celula.forcas)):
                        celula.forcas_velocidade[j] = (momentum * celula.forcas_velocidade[j] +
                            boost_factor * taxa_aprendizado * delta * saida_oculta[j])
                        celula.forcas[j] += celula.forcas_velocidade[j]
                    celula.tendencia_velocidade = momentum * celula.tendencia_velocidade + boost_factor * taxa_aprendizado * delta
                    celula.tendencia += celula.tendencia_velocidade

                # PASSO 3: Retropropagação para a camada oculta com boost_factor
                for i, celula in enumerate(self.camada_oculta.celulas):
                    erro_acumulado = 0
                    for k, celula_saida in enumerate(self.camada_saida.celulas):
                        erro_acumulado += celula_saida.forcas[i] * (
                            erros_saida[k] * pulse_activation_derivative(celula_saida.soma_entradas)
                        )
                    derivada = pulse_activation_derivative(celula.soma_entradas)
                    base_delta = erro_acumulado * derivada
                    delta = base_delta / (1 + abs(base_delta))  # estabiliza o delta
                    clip_value = 1.0  # gradient clipping
                    if delta > clip_value:
                        delta = clip_value
                    elif delta < -clip_value:
                        delta = -clip_value
                    for j in range(len(celula.forcas)):
                        celula.forcas_velocidade[j] = (momentum * celula.forcas_velocidade[j] +
                            boost_factor * taxa_aprendizado * delta * entradas[j])
                        celula.forcas[j] += celula.forcas_velocidade[j]
                    celula.tendencia_velocidade = momentum * celula.tendencia_velocidade + boost_factor * taxa_aprendizado * delta
                    celula.tendencia += celula.tendencia_velocidade

            print(f"Época {epoca + 1}/{epocas} - Erro Total (MSE): {erro_total:.4f}")

            # Estratégia adaptativa mais agressiva para a taxa de aprendizado
            if prev_error - erro_total < 0.001 * prev_error:
                taxa_aprendizado *= 1.05  # aumenta 5% se a melhora for pequena
            else:
                taxa_aprendizado *= 0.98  # diminui 2% se houver boa melhora
            prev_error = erro_total

            # Otimizador OscillaBoost: ajuste do parâmetro de modulação no transformer
            if self.transformer is not None:
                adjustment = math.sin(erro_total) * math.exp(-abs(erro_total))
                self.transformer.modulation += taxa_aprendizado * adjustment

    # Novo método para converter a saída em inteiros (mantido para compatibilidade)
    def para_inteiro(self, saida):
        # Converte cada elemento da saída para inteiro utilizando arredondamento
        return [int(round(s)) for s in saida]

    def gerar_resposta(self, prompt, max_steps=10):
        """
        Gera uma resposta de forma autoregressiva a partir de um prompt (sequência de tamanho fixo).
        Em cada step, utiliza a saída da rede para inferir o token com maior ativação e atualiza a sequência.
        Retorna a sequência final gerada.
        """
        response = prompt[:]  # Copia do prompt
        for _ in range(max_steps):
            pred = self.prever(response)
            # Aplica punição para repetição de tokens: tokens já presentes em 'response'
            # terão suas ativações penalizadas para reduzir repetições.
            freq = {}
            for tok in response:
                freq[tok] = freq.get(tok, 0) + 1
            penalized_pred = pred[:]  # Cópia das ativações
            penalty_factor = 0.5  # Fator de penalização por ocorrência
            for i in range(len(penalized_pred)):
                if i in freq:
                    penalized_pred[i] -= penalty_factor * freq[i]

            # Seleciona o token com maior ativação ajustada (penalizada)
            token = penalized_pred.index(max(penalized_pred))
            # Atualiza a sequência: remove o primeiro token e adiciona o novo token ao final
            response = response[1:] + [token]
        return response

    def melhorar_modelo(self, epocas, taxa_aprendizado=0.0000000005):
        """
        Método para melhorar (fine-tuning) o modelo com ajustes finos dos parâmetros.
        Executa uma série de iterações que aplicam pequenas alterações aleatórias nos pesos
        e vieses das células para otimizar o desempenho da rede.
        """
        momentum = 0.97
        boost_factor = 1.2
        for epoca in range(epocas):
            # Ajuste fino: pequenas perturbações aleatórias nos pesos e bias de cada célula
            for camada in [self.camada_oculta, self.camada_saida]:
                for celula in camada.celulas:
                    for i in range(len(celula.forcas)):
                        ajuste = boost_factor * taxa_aprendizado * random.uniform(-1.0, 1.0)
                        celula.forcas[i] += ajuste
                    celula.tendencia += boost_factor * taxa_aprendizado * random.uniform(-1.0, 1.0)
            # Ajuste adicional no módulo transformer, se integrado
            if self.transformer is not None:
                self.transformer.modulation += boost_factor * taxa_aprendizado * math.sin(epoca)
            # Exibe um aviso a cada 100 épocas para acompanhar o avanço
            if (epoca + 1) % 100 == 0:
                print(f"Melhoria {epoca + 1}/{epocas} executada.")

    def salvar_modelo(self, nome_arquivo):
        """
        Salva o modelo atual em um arquivo com o nome especificado.
        O modelo é serializado como um arquivo com extensão .nqn.
        """
        arquivo = nome_arquivo + ".nqn"
        with open(arquivo, "wb") as f:
            pickle.dump(self, f)
        print(f"Modelo salvo em {arquivo}")

    @classmethod
    def carregar_modelo(cls, nome_arquivo):
        """
        Carrega e retorna um modelo salvo a partir do arquivo especificado.
        """
        arquivo = nome_arquivo + ".nqn"
        with open(arquivo, "rb") as f:
            modelo = pickle.load(f)
        print(f"Modelo carregado de {arquivo}")
        return modelo

import math
import random
import pickle
import time
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
        self.tamanho_oculto = tamanho_oculto
        self.tamanho_saida = tamanho_saida
        self.transformer = None  # Novo: atributo para armazenar o módulo transformer integrado
        self.camada_oculta = Camada(tamanho_oculto, tamanho_entrada)
        self.camada_saida = Camada(tamanho_saida, tamanho_oculto)

    # Novo método para integrar um transformer (ex.: CosmicResonanceModulator)
    def integrar_transformer(self, transformer):
        self.transformer = transformer

    def prever(self, entradas):
        saida_oculta = self.camada_oculta.frente(entradas)
        if self.transformer is not None:
            saida_oculta = self.transformer.transform(saida_oculta)
        saida = self.camada_saida.frente(saida_oculta)
        return saida

    def treinar(self, dados_treinamento, epocas, taxa_aprendizado=0.0000000005, ciclos_melhoria=5, boost_factor=1.5):
        """
        Método de treinamento aprimorado com ciclos de melhoria integrados
        """
        for epoca in range(epocas):
            # Fase de treinamento padrão
            for entrada, saida_esperada in dados_treinamento:
                saida = self.prever(entrada)
                self.retropropagar(saida_esperada, taxa_aprendizado)
            
            # Fase de melhoria a cada ciclo
            if (epoca + 1) % (epocas//ciclos_melhoria) == 0:
                # Aumenta temporariamente a taxa de aprendizado
                self.melhorar_modelo(epocas=100, 
                                   taxa_aprendizado=taxa_aprendizado*boost_factor)
                
                print(f"Treinamento {epoca+1}/{epocas} completado com reforço")

    def retropropagar(self, saida_esperada, taxa_aprendizado):
        # Implemente a retropropagação para atualizar os pesos e vieses
        pass

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

    def gerar_resposta(self, prompt_tokens, max_steps, mostrar_tokens_atencao=False, 
                      tokenizer=None, prompt_text="", penalidade_repeticao=0.7):
        if self.transformer is None:
            raise Exception("Transformer não integrado. Use integrar_transformer para integrá-lo.")
        
        # Convert prompt tokens to initial activations
        current = [0.0] * len(tokenizer.vocab)
        for idx in prompt_tokens:
            if idx < len(current):
                current[idx] = 1.0  # Ativação inicial para tokens do prompt
        
        resposta = []
        tokens_atencao = []
        generated_token_ids = []
        
        for step in range(max_steps):
            # Aplica a transformação do módulo Cosmic
            current = self.transformer.transform(current)
            
            # Aplica penalidade de repetição
            logits = current.copy()
            for token_id in generated_token_ids:
                if token_id < len(logits):
                    logits[token_id] *= penalidade_repeticao
                    
            # Calcula probabilidades
            exp_logits = [math.exp(l) for l in logits]
            sum_exp = sum(exp_logits)
            probs = [e / sum_exp for e in exp_logits]
            
            # Seleciona o próximo token
            token_id = probs.index(max(probs))
            generated_token_ids.append(token_id)
            resposta.append(token_id)
            
            # Exibe informações de debug se necessário
            if mostrar_tokens_atencao:
                if tokenizer is None:
                    raise Exception("Tokenizador não fornecido.")
                
                # Prepara top 5
                top5 = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:5]
                
                print(f"\nPasso {step+1}:")
                print("Top 5 tokens:")
                for idx, prob in top5:
                    token = tokenizer.vocab[idx] if idx < len(tokenizer.vocab) else f"[{idx}]"
                    rep_flag = "(!REPETIDO!)" if idx in generated_token_ids else ""
                    print(f"'{token}' {rep_flag}: {prob*100:.2f}%")
                
                chosen_token = tokenizer.vocab[token_id] if token_id < len(tokenizer.vocab) else f"[{token_id}]"
                print(f"\nToken escolhido: {chosen_token}")
                print("="*50)
                time.sleep(1)

        if mostrar_tokens_atencao:
            return resposta, tokens_atencao
        return resposta

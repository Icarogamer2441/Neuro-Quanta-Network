import math
import random  # Adicionado para suportar jitter na fase

# Módulo CosmicResonanceModulator:
# Transforma as ativações aplicando modulação harmônica com variação de fase e combinação de funções senoidais e cosenoidais,
# promovendo uma ressonância cósmica que harmoniza os sinais da rede, diferenciando-a das arquiteturas feed-forward convencionais.
class CosmicResonanceModulator:
    def __init__(self, config='padrao', tem_atencao=False, camadas_atencao=1):
        if config == 'inovadora':
            self.modulation = 1.5  # Intensidade maior para acelerar as transformações
            self.phase = math.pi / 2  # Fase base elevada para modulação mais dinâmica
            self.jitter = 0.2         # Varredura de jitter aumentada para maior variabilidade
        else:
            self.modulation = 0.3
            self.phase = math.pi / 8
            self.jitter = 0.05
        # Armazena os novos parâmetros para atenção
        self.tem_atencao = tem_atencao
        self.camadas_atencao = camadas_atencao

    def transform(self, activations):
        # Calcula a média e o desvio padrão das ativações
        media = sum(activations) / len(activations)
        std = (sum((a - media) ** 2 for a in activations) / len(activations)) ** 0.5

        transformed = []
        for a in activations:
            # Calcula uma fase ajustada com jitter
            phase_adjusted = self.phase + random.uniform(-self.jitter, self.jitter)
            # Aplica modulação combinando funções senoidal e cosenoidal com decaimento exponencial
            modulated = a + self.modulation * (math.sin(a * phase_adjusted) + math.cos(a * phase_adjusted)) * math.exp(-abs(a))
            # Se o desvio padrão for muito pequeno (sistema preso), adiciona um impulso extra
            if std < 0.01:
                modulated += random.uniform(-0.5, 0.5)
            transformed.append(modulated)

        # Se o mecanismo de atenção estiver ativado, aplica uma camada simples de self-attention
        if self.tem_atencao:
            for _ in range(self.camadas_atencao):
                new_transformed = []
                # Para cada ativação, recalcula seu valor com base na similaridade com as demais
                for i in range(len(transformed)):
                    # Usa similaridade Gaussiana para calcular os pesos de atenção
                    weights = [math.exp(-((transformed[i] - transformed[j]) ** 2)) for j in range(len(transformed))]
                    weight_sum = sum(weights)
                    # Atualiza a ativação como média ponderada das ativações
                    new_value = sum(weights[j] * transformed[j] for j in range(len(transformed))) / weight_sum
                    new_transformed.append(new_value)
                transformed = new_transformed

        return transformed  # Retorna as ativações transformadas

    def generate_response(self, initial_state, steps):
        """
        Método simples para geração de resposta a partir de um estado inicial.
        Para cada step, aplica a transformação e seleciona o token com maior ativação (simulando uma função de argmax).
        Retorna uma sequência de índices.
        """
        current = initial_state[:]
        response = []
        for _ in range(steps):
            current = self.transform(current)
            # Seleciona o índice do maior valor como "token gerado"
            token_index = current.index(max(current))
            response.append(token_index)
        return response 
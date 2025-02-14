# NeuroQuantaNetwork

NeuroQuantaNetwork é um algoritmo de inteligência artificial inovador e simples, que visa reproduzir o comportamento neural de forma mais biológica e eficiente. Ele utiliza componentes únicos como **forças**, **tendências** e a função de ativação **PulseWave** para simular redes neurais que aprendem de maneira eficaz.

## Características Principais

- **NeuroQuanta** combina conceitos biológicos com uma matemática original e otimizada.
- Dispensa bibliotecas externas como NumPy, sendo implementado em Python puro.
- Utiliza uma função de ativação personalizada (**PulseWave**) que combina propriedades de `tanh` e `sin` para simular oscilações neurais.
- Substitui conceitos tradicionais como "weights" e "bias" por **forças** e **tendências**, tornando o sistema mais simples e inspirado no funcionamento do cérebro humano.

---

## Estrutura do Projeto

### **Funções e Componentes**

#### 1. **Função de Ativação - PulseWave**
```python
pulse_activation(x)
```
- Combina `tanh(x)` e `sin(x)` para criar uma função de ativação não linear e dinâmica.
- Retorna o valor de ativação com base na soma ponderada dos inputs.

#### 2. **Derivada da Função de Ativação**
```python
pulse_activation_derivative(x)
```
- Retorna a derivada aproximada da função `PulseWave`, usada para ajustes durante o treinamento.

#### 3. **Célula (Celula)**
```python
class Celula:
    def __init__(self, num_entradas):
        ...
```
- Representa um único "neurônio" ou unidade de processamento.
- **forcas:** Representam as influências de cada entrada (equivalente aos pesos tradicionais).
- **tendencia:** Atua como um deslocamento que ajusta o comportamento do neurônio.
- **frente(entradas):** Realiza o cálculo da soma ponderada e aplica a função de ativação para produzir a saída.

#### 4. **Camada (Camada)**
```python
class Camada:
    def __init__(self, num_celulas, num_entradas_por_celula):
        ...
```
- Uma camada composta por várias células (neurônios).
- **frente(entradas):** Propaga os sinais de entrada através da camada e retorna as saídas de todas as células.

#### 5. **Rede Neural (NeuroQuantaNetwork)**
```python
class NeuroQuantaNetwork:
    def __init__(self, tamanho_entrada, tamanho_oculto, tamanho_saida):
        ...
```
- Estrutura principal da rede.
- Possui uma camada oculta e uma camada de saída.
- **prever(entradas):** Realiza a propagação direta das entradas para gerar uma previsão.
- **treinar(dados_treinamento, epocas, taxa_aprendizado):** Executa o treinamento da rede usando os dados fornecidos.

## Módulo de Ressonância Cósmica: CosmicResonanceModulator e Otimizador OscillaBoost

O NeuroQuantaNetwork agora conta com um módulo transformer inovador e original, denominado **CosmicResonanceModulator**. Este sistema foi criado para potencializar o processamento de informações sequenciais e dinâmicas, utilizando uma matemática exclusiva que combina variações angulares e transformações de "fluxo de quanta", elevando a capacidade adaptativa da rede.

### CosmicResonanceModulator
- **Nome:** CosmicResonanceModulator.
- **Função:** Realiza transformações dinâmicas das ativações internas aplicando modulação harmônica, jitter de fase e decaimento exponencial, promovendo uma ressonância cósmica que harmoniza os sinais da rede.
- **Matemática Inovadora:** Combina funções senoidais, cosenoidais e exponenciais para ajustar as ativações de forma única, diferenciando-se dos modelos feed-forward convencionais.

### Otimizador OscillaBoost
- **Nome Original:** OscillaBoost.
- **Função:** Otimiza os parâmetros do CosmicResonanceModulator (e, consequentemente, da NeuroQuantaNetwork) através de ajustes proporcionais baseados em oscilações angulares e exponenciais.
- **Matemática Original:** Utiliza um mecanismo de ajuste que incorpora ondas senoidais moduladas por funções exponenciais, promovendo uma convergência mais rápida e robusta durante o treinamento.

### Integração Exclusiva com a NeuroQuanta Network
Este módulo transformer foi desenvolvido para ser integrado **exclusivamente** à NeuroQuantaNetwork. Ao utilizar o CosmicResonanceModulator, a rede expande sua capacidade de processamento e adaptação, tornando-se ainda mais eficaz em lidar com dados sequenciais e temporais.

#### Exemplo de Uso com CosmicResonanceModulator e Otimizador OscillaBoost
```python
from neuroquanta import NeuroQuantaNetwork, CosmicResonanceModulator

# Configuração da NeuroQuanta com integração do CosmicResonanceModulator
rede = NeuroQuantaNetwork(tamanho_entrada=2, tamanho_oculto=4, tamanho_saida=1)

# Integra o módulo CosmicResonanceModulator à rede (método exclusivo da NeuroQuantaNetwork)
rede.integrar_transformer(CosmicResonanceModulator(config='inovadora'))

# Dados de treinamento para a função XOR
dados_treinamento = [
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [0])
]

# Treina a rede utilizando o otimizador OscillaBoost interno
rede.treinar(dados_treinamento, epocas=1000, taxa_aprendizado=0.05)
```

---

## Como Usar

### 1. Requisitos
- Python 3.7 ou superior.
- Nenhuma biblioteca externa é necessária.

### 2. Exemplo de Uso

#### Criando a Rede
```python
from neuroquanta import NeuroQuantaNetwork

# Configuração da rede
rede = NeuroQuantaNetwork(tamanho_entrada=2, tamanho_oculto=4, tamanho_saida=1)
```

#### Dados de Treinamento
```python
# Dados de treinamento para a função XOR
dados_treinamento = [
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [0])
]
```

#### Treinando a Rede
```python
# Treinamento por 1000 épocas com taxa de aprendizado 0.05
rede.treinar(dados_treinamento, epocas=1000, taxa_aprendizado=0.05)
```

#### Testando a Rede
```python
# Testa os resultados
for entrada, esperado in dados_treinamento:
    saida = rede.prever(entrada)
    print(f"Entrada: {entrada} | Saída prevista: {saida} | Esperado: {esperado}")
```

### 3. Ajustando Parâmetros
- **tamanho_oculto:** Número de células na camada oculta.
- **epocas:** Número de iterações de treinamento.
- **taxa_aprendizado:** Define a rapidez com que os parâmetros são ajustados durante o treinamento.

---

## Matemática do NeuroQuanta

1. **Cálculo da Soma Ponderada**
   - Para cada célula, calcula-se:
     
     \[ \text{soma\_ponderada} = \sum (\text{entrada}_i \times \text{forca}_i) + \text{tendencia} \]

2. **Função de Ativação - PulseWave**
   - Combina `tanh` e `sin`:
     
     \[ \text{ativacao} = \frac{\tanh(x) + \sin(x)}{2} \]

3. **Atualização dos Parâmetros**
   - Ajusta as **forças** e **tendência**:
     
     \[ \Delta \text{forca}_i = \text{taxa\_aprendizado} \times \text{erro} \times \text{entrada}_i \]
     
     \[ \Delta \text{tendencia} = \text{taxa\_aprendizado} \times \text{erro} \]

---

## Licença
Este projeto é fornecido sob a licença MIT. Sinta-se à vontade para usar, modificar e distribuir conforme necessário.

---

## Contribuições
Contribuições para melhorar ou expandir o NeuroQuanta são bem-vindas! Caso tenha sugestões, abra uma _issue_ ou envie um _pull request_.

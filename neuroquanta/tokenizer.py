import re

class Tokenizer:
    def __init__(self, minusculas=False, usar_espacos=True):
        self.minusculas = minusculas
        self.usar_espacos = usar_espacos
        self.vocab = ["<PAD>"]  # Token de preenchimento no índice 0
        self.max_len = 0        # Comprimento máximo das sequências

    def _split_tokens(self, texto):
        # Se minusculas estiver ativo, converte para lower case
        if self.minusculas:
            texto = texto.lower()
        # Utiliza uma regex que separa:
        # - Sequências de caracteres alfanuméricos (palavras)
        # - Cada caractere que não for alfanumérico (ex.: pontuações, espaços)
        tokens = re.findall(r'\w+|[^\w]', texto)
        return tokens

    def adicionar(self, lista_textos):
        """
        Processa uma lista de textos, atualiza o vocabulário e guarda o comprimento máximo.
        """
        for texto in lista_textos:
            tokens = self._split_tokens(texto)
            # Atualiza o comprimento máximo se necessário
            if len(tokens) > self.max_len:
                self.max_len = len(tokens)
            # Adiciona cada token único (respeitando a ordem de aparecimento)
            for token in tokens:
                if token not in self.vocab:
                    self.vocab.append(token)

    @property
    def tam_vocabulario(self):
        return len(self.vocab)

    def tokenizar(self, texto):
        """
        Converte o texto de entrada em uma sequência fixa de índices (tamanho = self.max_len).
        Cada token é mapeado para seu índice no vocabulário (a ordem dos tokens é preservada).
        Se a sequência for menor que self.max_len, preenche com 0 (PAD).
        """
        tokens = self._split_tokens(texto)
        seq = []
        for token in tokens:
            try:
                idx = self.vocab.index(token)
            except ValueError:
                idx = 0  # fallback para PAD (não esperado)
            seq.append(idx)
        # Preenche (pad) se necessário para garantir tamanho fixo
        if len(seq) < self.max_len:
            seq += [0] * (self.max_len - len(seq))
        return seq

    def para_texto(self, seq):
        # Atualizado para lidar com listas aninhadas (ex.: tokens de atenção)
        if seq and isinstance(seq[0], list):
            lines = []
            for sub_seq in seq:
                indices = [int(round(x)) for x in sub_seq]
                # Procura o token correspondente, se o índice for válido; caso contrário, usa a representação numérica
                tokens = [self.vocab[i] if i < len(self.vocab) else str(i) for i in indices]
                lines.append(" ".join(tokens))
            return "\n".join(lines)
        else:
            indices = [int(round(x)) for x in seq]
            tokens = [self.vocab[i] if i < len(self.vocab) else str(i) for i in indices]
            return " ".join(tokens)

    def sequence_to_tokens(self, seq):
        """
        Método auxiliar para converter uma sequência de índices para a lista de tokens.
        """
        indices = [int(round(x)) for x in seq]
        return [self.vocab[idx] for idx in indices if idx != 0]

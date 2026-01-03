class BPE:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.id2token = {}
        self.token2id = {}

    def fit(self, text: str):
        tokens = list(text)

        unique_tokens = sorted(set(tokens))


        def get_pair_stats_and_first_pos(seq):
            counts = {}
            first_pos = {}
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i+1])
                counts[pair] = counts.get(pair, 0) + 1
                if pair not in first_pos:
                    first_pos[pair] = i
            return counts, first_pos

        def merge_all(seq, pair):
            a, b = pair
            merged = a + b
            out = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == a and seq[i+1] == b:
                    out.append(merged)
                    i += 2
                else:
                    out.append(seq[i])
                    i += 1
            return out, merged

        while len(unique_tokens) < self.vocab_size:
            counts, first_pos = get_pair_stats_and_first_pos(tokens)
            if not counts:
                break

            best_pair = max(counts.keys(), key=lambda p: (counts[p], -first_pos[p]))

            tokens, new_tok = merge_all(tokens, best_pair)
            unique_tokens.append(new_tok)

        self.id2token = {i: tok for i, tok in enumerate(unique_tokens[:self.vocab_size])}
        self.token2id = {tok: i for i, tok in self.id2token.items()}

        return unique_tokens[:self.vocab_size]

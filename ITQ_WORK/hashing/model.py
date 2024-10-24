

class Model():
    "Hashing model base."
    def __init__(self, encode_len):
        self.encode_len = encode_len

    def fit(self):
        raise NotImplementedError

    def encode(self):
        raise NotImplementedError

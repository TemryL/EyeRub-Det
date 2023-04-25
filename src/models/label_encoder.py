class LabelEncoder():
    def __init__(self):
        self.encode_map = {
            'Eating': 0,
            'Face touching': 0,
            'Eye rubbing': 1,
            'Eye rubbing light': 1,
            'Eye rubbing moderate': 1,
            'Eye touching': 0,
            'Glasses readjusting': 0,
            'Hair combing': 2,
            'Make up': 0,
            'Make up application': 0,
            'Make up removal': 0,
            'Skin scratching': 2,
            'Teeth brushing': 3,
            'Nothing': 4,
            'no_label': 4
        }
        self.decode_map = {
            0: "Face touching",
            1: "Eye rubbing",
            2: "Hair combing/Skin scratching",
            3: "Teeth brushing",
            4: "Nothing"
        }
    
    def transform(self, labels):
        return [self.encode_map[label] for label in labels]
    
    def inv_transform(self, labels):
        return [self.decode_map[label] for label in labels]
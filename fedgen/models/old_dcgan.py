# Minimal placeholders; replace with real torch modules.
class Generator64:
    def __init__(self, z_dim=128): self.z_dim=z_dim
    def state_dict(self): return {"w": 0.0}
    def load_state_dict(self, d): pass

class Discriminator64:
    def __init__(self): pass

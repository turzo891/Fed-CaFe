# Stub: hook in a pretrained race classifier for scoring generated images.
class RaceClassifier:
    def __init__(self, labels=("raceA","raceB","raceC","raceD")):
        self.labels = labels
    def predict(self, imgs): 
        # return dummy labels same length as imgs
        return [self.labels[0]]*len(imgs)

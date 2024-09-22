from sklearn.metrics import average_precision_score

class Evaluator:
    def __init__(self):
        pass

    def evaluate(self, predictions, true_labels):
        return average_precision_score(true_labels, predictions)

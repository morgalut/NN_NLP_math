from sklearn.metrics.pairwise import cosine_similarity

class SimilarityCalculator:
    def __init__(self):
        pass

    def compute_similarity(self, question_embeddings, misconception_embeddings):
        return cosine_similarity(question_embeddings, misconception_embeddings)

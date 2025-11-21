class NaiveBayesClassifier:
    def __init__(self):
        self.priors = {}
        self.conditionals = {}
        self.vocabulary = []

    @staticmethod
    def get_vocabulary(data):
        vocab_set = set()
        for doc in data:
            for word in doc[0]:
                vocab_set.add(word)

        return sorted(list(vocab_set))

    @staticmethod
    def _calculate_priors(data):
        """
        Internal method to calculate Prior probabilities: P(c) and P(j).
        """
        total_docs = len(data)
        c_count = 0
        j_count = 0

        for doc in data:
            match doc[1]:
                case 'c':
                    c_count += 1
                case 'j':
                    j_count += 1

        return {
            'c': c_count / total_docs,
            'j': j_count / total_docs
        }

    def _calculate_conditionals(self, data):
        """
        Internal method to calculate Conditional probabilities (Likelihoods): P(w|c).
        Using Laplace Smoothing (Add-1).
        """

        c_list = []
        j_list = []
        for doc in data:
            match doc[1]:
                case 'c':
                    c_list.extend(doc[0])
                case 'j':
                    j_list.extend(doc[0])

        # Calculate denominators (Total words in class + Vocabulary size)
        denom_c = len(c_list) + len(self.vocabulary)
        denom_j = len(j_list) + len(self.vocabulary)

        conditionals = {'c': {}, 'j': {}}

        # Calculate probability for each word in the extracted vocabulary
        for word in self.vocabulary:
            conditionals['c'][word] = (c_list.count(word) + 1) / denom_c
            conditionals['j'][word] = (j_list.count(word) + 1) / denom_j

        return conditionals

    def train(self, training_data):
        """
        Main training method:
        1. Automatically extracts vocabulary from data.
        2. Calculate Priors and Conditionals.
        """

        self.vocabulary = self.get_vocabulary(training_data)

        self.priors = self._calculate_priors(training_data)
        self.conditionals = self._calculate_conditionals(training_data)

    def predict(self, doc):
        """
        Prediction method: Takes a new document list and returns the predicted class.
        """
        score_c = self.priors['c']
        score_j = self.priors['j']

        for word in doc:
            if word in self.conditionals['c']:
                score_c *= self.conditionals['c'][word]
                score_j *= self.conditionals['j'][word]

        winner = 'c' if score_c > score_j else 'j'

        return {
            'predicted_class': winner,
            'score_c': score_c,
            'score_j': score_j
        }

training_data = [
    (['Chinese', 'Beijing', 'Chinese'], 'c'),
    (['Chinese', 'Chinese', 'Shanghai'], 'c'),
    (['Chinese', 'Macao'], 'c'),
    (['Tokyo', 'Japan', 'Chinese'], 'j'),
]
test_doc = ['Chinese', 'Tokyo', 'Shanghai']

model = NaiveBayesClassifier()

model.train(training_data)
result = model.predict(test_doc)

print("\n📊 Prediction Result:")
print(f"   Score Class 'c': {result['score_c']}")
print(f"   Score Class 'j': {result['score_j']}")
print(f"🏆 Winner: Class '{result['predicted_class']}'")
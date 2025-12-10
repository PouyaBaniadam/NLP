import math


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
    def _calculate_priors(train_data):
        total_docs = len(train_data)
        c_count = 0
        j_count = 0

        for doc, category in train_data:
            match category:
                case 'c':
                    c_count += 1
                case 'j':
                    j_count += 1

        return {
            'c': c_count / total_docs,
            'j': j_count / total_docs,
        }

    # Task: Calculate conditional probabilities, P(word|class), using Laplace Smoothing.
    def _calculate_conditionals(self, training_data):
        c_list = []
        j_list = []
        for doc, category in training_data:
            match category:
                case 'c':
                    c_list.extend(doc)
                case 'j':
                    j_list.extend(doc)

        # Calculate the denominator for the Laplace Smoothing formula.
        # Denominator = (Total number of words in the class) + (Size of the vocabulary)
        denom_c = len(c_list) + len(self.vocabulary)  # For class 'c': 8 + 6 = 14
        denom_j = len(j_list) + len(self.vocabulary)  # For class 'j': 3 + 6 = 9

        conditionals = {'c': {}, 'j': {}}

        for word in self.vocabulary:
            conditionals['c'][word] = (c_list.count(word) + 1) / denom_c
            conditionals['j'][word] = (j_list.count(word) + 1) / denom_j

        return conditionals

    def train(self, training_data):
        self.vocabulary = self.get_vocabulary(training_data)
        self.priors = self._calculate_priors(training_data)
        self.conditionals = self._calculate_conditionals(training_data)

    def predict(self, to_be_predicted_doc):
        score_c = self.priors['c']  # Starts at 0.75
        score_j = self.priors['j']  # Starts at 0.25

        for word in to_be_predicted_doc:
            score_c *= self.conditionals['c'][word]
            score_j *= self.conditionals['j'][word]

        winner = 'c' if score_c > score_j else 'j'

        return {
            'predicted_class': winner,
            'score_c': math.log(score_c),
            'score_j': math.log(score_j)
        }


training_data = [
    (['Chinese', 'Beijing', 'Chinese'], 'c'),
    (['Chinese', 'Chinese', 'Shanghai'], 'c'),
    (['Chinese', 'Macao'], 'c'),
    (['Tokyo', 'Japan', 'Chinese'], 'j'),
]

to_be_predicted_doc = ['Chinese', 'Tokyo', 'Shanghai']


model = NaiveBayesClassifier()

model.train(training_data)

result = model.predict(to_be_predicted_doc)

print("\n📊 Prediction Result:")
print(f"   Score Class 'c': {result['score_c']}")
print(f"   Score Class 'j': {result['score_j']}")
print(f"🏆 Winner: Class '{result['predicted_class']}'")
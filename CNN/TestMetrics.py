from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

class TestMetrics:
    def __init__(self, true_labels, predicted_labels):
        self.true_labels = true_labels
        self.predicted_labels = predicted_labels
        self.confusion_matrix = confusion_matrix(true_labels, predicted_labels)

    def calculate_metrics(self):
        precision = precision_score(self.true_labels, self.predicted_labels, average=None)
        recall = recall_score(self.true_labels, self.predicted_labels, average=None)
        f1 = f1_score(self.true_labels, self.predicted_labels, average=None)
        return precision, recall, f1

    def weighted_metrics(self):
        weighted_precision = precision_score(self.true_labels, self.predicted_labels, average='weighted')
        weighted_recall = recall_score(self.true_labels, self.predicted_labels, average='weighted')
        weighted_f1 = f1_score(self.true_labels, self.predicted_labels, average='weighted')
        return weighted_precision, weighted_recall, weighted_f1

    def print_metrics(self):
        print('Confusion Matrix:')
        print(self.confusion_matrix)
        precision, recall, f1 = self.calculate_metrics()
        print('Precision:')
        print(precision)
        print('Recall:')
        print(recall)
        print('F1 score:')
        print(f1)
        weighted_precision, weighted_recall, weighted_f1 = self.weighted_metrics()
        print(f'Weighted precision: {weighted_precision:.4f}')
        print(f'Weighted recall: {weighted_recall:.4f}')
        print(f'Weighted F1 score: {weighted_f1:.4f}')

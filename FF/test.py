import numpy as np
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter

wartosci = [
    [41.7386458623572, 40.59626636946225, 39.314572304263024, 40.930621342992474],
    [35.21872387851769, 42.63025912510449, 40.37336305377543, 40.01114516578434],
    [38.06074115352466, 37.55920869322931, 41.12566174421845, 40.20618556701031],
    [40.0947339091669, 36.97408748955141, 37.55920869322931, 40.98634717191418]
]

learning_rateArray = [0.0005,0.001,0.0015, 0.002]
batchSizeArray = [16,32,64,128]

plt.imshow(wartosci, cmap='viridis')
plt.title("Barch norm and Learning rate testing")
plt.colorbar()  # Dodaje pasek kolorów dla wartości
plt.xticks(np.arange(len(learning_rateArray)), learning_rateArray)
plt.yticks(np.arange(len(batchSizeArray)), batchSizeArray)
plt.xlabel("Learning rate")
plt.ylabel("Batch norm")
for (i, j), z in np.ndenumerate(wartosci):
    plt.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

plt.show()

training_class = np.array([0,0,0,0,1])
predicted_train_class = np.array([0,1,0,1,1])

testing_class = np.array([0,0,0,0,1])
predicted_test_class = np.array([0,1,0,1,1])

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
labels = ['happy', 'sad']
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
#    3.1 Calculate the confusion matrix when classifying the training data

ax[0].title.set_text('Training data confusion matrix:')
cm = confusion_matrix(training_class.ravel(),predicted_train_class)
cmd = ConfusionMatrixDisplay(cm, display_labels=['happy', 'sad'])
cmd.plot(ax=ax[0])


#    3.2 Calculate the confusion matrix when classifying the testing data
ax[1].title.set_text('Testing data confusion matrix:')
cm2 = confusion_matrix(testing_class.ravel(),predicted_test_class)
cmd2 = ConfusionMatrixDisplay(cm2, display_labels=['happy', 'sad'])
cmd2.plot(ax=ax[1])
plt.show()
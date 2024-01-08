import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from google.colab import files
import iotrain_labels = train_data['5']
train_pixels = train_data.drop('5', axis=1)

test_labels = test_data['7']
test_pixels = test_data.drop('7', axis=1)from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=True)

# Separate features (X) and target (y)
X = mnist['data']
y = mnist['target']
# Split the data into train and test sets
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# Initialize the random forest classifier
clf = RandomForestClassifier(n_estimators=100)

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print('The accuracy of the random forest classifier is:', accuracy)

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
data_train = pd.read_csv('mnist_train.csv')
data_test = pd.read_csv('mnist_test.csv')

X_train = data_train.iloc[:, 1:].values
y_train = data_train.iloc[:, 0].values

X_test = data_test.iloc[:, 1:].values
y_test = data_test.iloc[:, 0].values

image_index = 0
image = X_train[image_index].reshape((28, 28))

plt.imshow(image, cmap='gray')
plt.title(f"Label: {y_train[image_index]}")
plt.axis('off')
plt.show()
data_train = pd.read_csv('mnist_train.csv')
data_test = pd.read_csv('mnist_test.csv')

X_train = data_train.iloc[:59999, 1:]
y_train = data_train.iloc[:59999, 0]

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train)

df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
df_pca['label'] = y_train

plt.figure(figsize=(6, 4))
scatter = plt.scatter(df_pca['PC1'], df_pca['PC2'], c=df_pca['label'], cmap='tab10', alpha=0.8)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('MNIST Data Visualization using PCA')
plt.legend(handles=scatter.legend_elements()[0], labels=range(10), title='Digit')
plt.show()
data_train = pd.read_csv('mnist_train.csv')
data_test = pd.read_csv('mnist_test.csv')
X_train = data_train.iloc[:, 1:]
y_train = data_train.iloc[:, 0]
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

feature_importance = clf.feature_importances_
feature_names = X_train.columns
sorted_indices = feature_importance.argsort()[::-1]
sorted_feature_importance = feature_importance[sorted_indices]
sorted_feature_names = feature_names[sorted_indices]


plt.figure(figsize=(8, 4))
plt.bar(range(len(feature_importance)), sorted_feature_importance, tick_label=sorted_feature_names)
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.title('Feature Importance')
plt.tight_layout()
plt.show()
# Set the hyperparameters
learning_rate = 0.01
regularization_weight = 1
num_iterations = 100

# Train the logistic regression classifier
beta, costs, train_accs, test_accs = logistic_regression(train_pixels, train_labels, learning_rate, regularization_weight, num_iterations)

# Plot the cost value, training accuracy, and testing accuracy versus the number of iterations
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(num_iterations), costs)
plt.xlabel('Number of Iterations')
plt.ylabel('Cost Value')
plt.title('Cost Value vs. Number of Iterations')

plt.subplot(1, 2, 2)
plt.plot(range(num_iterations), train_accs, label='Training')
plt.plot(range(num_iterations), test_accs, label='Testing')
for regularization_weight in regularization_weights:
    beta, _, _, test_accs = logistic_regression(train_pixels, train_labels, learning_rate, regularization_weight, num_iterations)
    final_test_accuracy = test_accs[-1]
    results_table.append([regularization_weight, final_test_accuracy])

# Print the results table
print("Regularization Weight\tFinal Testing Accuracy")
for row in results_table:
    print(f"{row[0]}\t\t\t{row[1]}")


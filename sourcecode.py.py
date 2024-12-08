

# Install necessary libraries
# !pip install scikit-learn
# !pip install pandas numpy matplotlib seaborn

import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

column_names = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
                'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
                'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
                'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login',
                'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
                'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
                'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
                'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label']

data = pd.read_csv(r"C:\Users\P.rashmitha\Downloads\Exec\kddcup.data_10_percent", header=None, names=column_names)

"""# Data Cleansing and Preparation"""

# Remove duplicates
data.drop_duplicates(inplace=True)

# Encode categorical features
categorical_features = ['protocol_type', 'service', 'flag']
data = pd.get_dummies(data, columns=categorical_features)

# Convert labels to binary (0 for normal, 1 for anomaly)
data['label'] = data['label'].apply(lambda x: 0 if x == 'normal.' else 1)

# For simplicity, we will use only numerical features
data_features = data.drop(columns=['label'])
labels = data['label']

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
data_features_imputed = imputer.fit_transform(data_features)
data_features_imputed = pd.DataFrame(data_features_imputed, columns=data_features.columns)

print("Checking for NaN values after imputation:")
print(data_features_imputed.isna().sum().sum())



# Normalize data
data_features_imputed = (data_features_imputed - data_features_imputed.mean()) / data_features_imputed.std()

# Ensure no NaN values after normalization
data_features_imputed.fillna(0, inplace=True)

# Reset index to ensure alignment
data_features_imputed.reset_index(drop=True, inplace=True)
labels.reset_index(drop=True, inplace=True)

# Let's randomly sample a subset for visualization purposes
sample_indices = np.random.choice(data_features_imputed.index, size=1000, replace=False)
data_sampled = data_features_imputed.loc[sample_indices]
labels_sampled = labels.loc[sample_indices]

"""#Plotting the outliers"""

# Function to plot data with outliers
def plot_outliers(data, outliers, title, x_label='Feature 1', y_label='Feature 2'):
    plt.figure(figsize=(10, 6))
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c='blue', label='Normal')
    plt.scatter(data.iloc[outliers, 0], data.iloc[outliers, 1], c='red', label='Outliers')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()

"""# 1. Local Outlier Factor"""

lof = LocalOutlierFactor(n_neighbors=20)
outliers_lof = lof.fit_predict(data_sampled)
outliers_lof = np.where(outliers_lof == -1, 1, 0)
accuracy_lof = accuracy_score(labels_sampled, outliers_lof)
precision_lof = precision_score(labels_sampled, outliers_lof)
recall_lof = recall_score(labels_sampled, outliers_lof)
f1_lof = f1_score(labels_sampled, outliers_lof)
plot_outliers(data_sampled, np.where(outliers_lof == 1)[0], f'Local Outlier Factor (LOF), Accuracy: {accuracy_lof:.2f}')

"""# 2. K-Nearest Neighbors (KNN)"""

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(data_sampled, labels_sampled)
distances, _ = knn.kneighbors(data_sampled)
outliers_knn = np.where(distances.mean(axis=1) > np.percentile(distances.mean(axis=1), 95), 1, 0)
accuracy_knn = accuracy_score(labels_sampled, outliers_knn)
precision_knn = precision_score(labels_sampled, outliers_knn)
recall_knn = recall_score(labels_sampled, outliers_knn)
f1_knn = f1_score(labels_sampled, outliers_knn)
plot_outliers(data_sampled, np.where(outliers_knn == 1)[0], f'K-Nearest Neighbors (KNN), Accuracy: {accuracy_knn:.2f}')

"""# 3. One-Class SVM"""

ocsvm = OneClassSVM(gamma='auto')
outliers_ocsvm = ocsvm.fit_predict(data_sampled)
outliers_ocsvm = np.where(outliers_ocsvm == -1, 1, 0)
accuracy_ocsvm = accuracy_score(labels_sampled, outliers_ocsvm)
precision_ocsvm = precision_score(labels_sampled, outliers_ocsvm)
recall_ocsvm = recall_score(labels_sampled, outliers_ocsvm)
f1_ocsvm = f1_score(labels_sampled, outliers_ocsvm)
plot_outliers(data_sampled, np.where(outliers_ocsvm == 1)[0], f'One-Class SVM, Accuracy: {accuracy_ocsvm:.2f}')

"""# 4. Isolation Forest"""

iforest = IsolationForest(contamination=0.1)
outliers_iforest = iforest.fit_predict(data_sampled)
outliers_iforest = np.where(outliers_iforest == -1, 1, 0)
accuracy_iforest = accuracy_score(labels_sampled, outliers_iforest)
precision_iforest = precision_score(labels_sampled, outliers_iforest)
recall_iforest = recall_score(labels_sampled, outliers_iforest)
f1_iforest = f1_score(labels_sampled, outliers_iforest)
plot_outliers(data_sampled, np.where(outliers_iforest == 1)[0], f'Isolation Forest, Accuracy: {accuracy_iforest:.2f}')

"""# 5. K-Means Clustering"""

kmeans = KMeans(n_clusters=10)
clusters = kmeans.fit_predict(data_sampled)
distances = kmeans.transform(data_sampled).min(axis=1)
outliers_kmeans = np.where(distances > np.percentile(distances, 95), 1, 0)
accuracy_kmeans = accuracy_score(labels_sampled, outliers_kmeans)
precision_kmeans = precision_score(labels_sampled, outliers_kmeans)
recall_kmeans = recall_score(labels_sampled, outliers_kmeans)
f1_kmeans = f1_score(labels_sampled, outliers_kmeans)
plot_outliers(data_sampled, np.where(outliers_kmeans == 1)[0], f'K-Means Clustering, Accuracy: {accuracy_kmeans:.2f}')

"""# 6. DBSCAN"""

dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(data_sampled)
outliers_dbscan = np.where(clusters == -1, 1, 0)
accuracy_dbscan = accuracy_score(labels_sampled, outliers_dbscan)
precision_dbscan = precision_score(labels_sampled, outliers_dbscan)
recall_dbscan = recall_score(labels_sampled, outliers_dbscan)
f1_dbscan = f1_score(labels_sampled, outliers_dbscan)
plot_outliers(data_sampled, np.where(outliers_dbscan == 1)[0], f'DBSCAN, Accuracy: {accuracy_dbscan:.2f}')

"""# Print accuracy scores"""

print(f'Accuracy Scores:')
print(f'Local Outlier Factor (LOF): {accuracy_lof:.2f}')
print(f'K-Nearest Neighbors (KNN): {accuracy_knn:.2f}')
print(f'One-Class SVM: {accuracy_ocsvm:.2f}')
print(f'Isolation Forest: {accuracy_iforest:.2f}')
print(f'K-Means Clustering: {accuracy_kmeans:.2f}')
print(f'DBSCAN: {accuracy_dbscan:.2f}')

# Print precision, recall, F1 scores
print(f'Precision Scores:')
print(f'Local Outlier Factor (LOF): {precision_lof:.2f}')
print(f'K-Nearest Neighbors (KNN): {precision_knn:.2f}')
print(f'One-Class SVM: {precision_ocsvm:.2f}')
print(f'Isolation Forest: {precision_iforest:.2f}')
print(f'K-Means Clustering: {precision_kmeans:.2f}')
print(f'DBSCAN: {precision_dbscan:.2f}')

print(f'Recall Scores:')
print(f'Local Outlier Factor (LOF): {recall_lof:.2f}')
print(f'K-Nearest Neighbors (KNN): {recall_knn:.2f}')
print(f'One-Class SVM: {recall_ocsvm:.2f}')
print(f'Isolation Forest: {recall_iforest:.2f}')
print(f'K-Means Clustering: {recall_kmeans:.2f}')
print(f'DBSCAN: {recall_dbscan:.2f}')

print(f'F1 Scores:')
print(f'Local Outlier Factor (LOF): {f1_lof:.2f}')
print(f'K-Nearest Neighbors (KNN): {f1_knn:.2f}')
print(f'One-Class SVM: {f1_ocsvm:.2f}')
print(f'Isolation Forest: {f1_iforest:.2f}')
print(f'K-Means Clustering: {f1_kmeans:.2f}')
print(f'DBSCAN: {f1_dbscan:.2f}')

# Plot comparison of the metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
lof_scores = [accuracy_lof, precision_lof, recall_lof, f1_lof]
knn_scores = [accuracy_knn, precision_knn, recall_knn, f1_knn]
ocsvm_scores = [accuracy_ocsvm, precision_ocsvm, recall_ocsvm, f1_ocsvm]
iforest_scores = [accuracy_iforest, precision_iforest, recall_iforest, f1_iforest]
kmeans_scores = [accuracy_kmeans, precision_kmeans, recall_kmeans, f1_kmeans]
dbscan_scores = [accuracy_dbscan, precision_dbscan, recall_dbscan, f1_dbscan]

scores = pd.DataFrame({
    'Metric': metrics,
    'LOF': lof_scores,
    'KNN': knn_scores,
    'One-Class SVM': ocsvm_scores,
    'Isolation Forest': iforest_scores,
    'K-Means': kmeans_scores,
    'DBSCAN': dbscan_scores
})

scores.set_index('Metric', inplace=True)

scores.plot(kind='bar', figsize=(14, 8), colormap='viridis')
plt.title('Comparison of Outlier Detection Methods')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.legend(loc='best')
plt.show()


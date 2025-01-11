import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score

fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)


autoencoder = load_model("autoencoder_best_model.keras")
encoder = tf.keras.Model(inputs=autoencoder.input, outputs=autoencoder.layers[-7].output)  


train_embeddings = encoder.predict(x_train)
test_embeddings = encoder.predict(x_test)


train_embeddings_flat = train_embeddings.reshape(len(train_embeddings), -1)
test_embeddings_flat = test_embeddings.reshape(len(test_embeddings), -1)


def evaluate_similarity_search():
    precision_list = []
    recall_list = []
    total_retrievals = 0
    correct_retrievals = 0

    for i in range(len(x_test)):  
        query_image = x_test[i]
        true_label = y_test[i]

        
        query_embedding = encoder.predict(query_image.reshape(1, 28, 28, 1)).reshape(1, -1)
        
        
        similarities = cosine_similarity(query_embedding, train_embeddings_flat)
        most_similar_indices = np.argsort(similarities[0])[::-1][:5]  
        similar_labels = y_train[most_similar_indices]

        
        precision = precision_score([true_label] * len(similar_labels), similar_labels, average='micro', zero_division=1)
        recall = recall_score([true_label] * len(similar_labels), similar_labels, average='micro', zero_division=1)
        
        precision_list.append(precision)
        recall_list.append(recall)
        
        
        if true_label in similar_labels:
            correct_retrievals += 1
        total_retrievals += 1

    
    mean_precision = np.mean(precision_list)
    mean_recall = np.mean(recall_list)
    retrieval_accuracy = correct_retrievals / total_retrievals

    print(f"Precision: {mean_precision:.4f}")
    print(f"Recall: {mean_recall:.4f}")
    print(f"Retrieval Accuracy: {retrieval_accuracy:.4f}")


evaluate_similarity_search()

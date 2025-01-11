import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

autoencoder = load_model("autoencoder_final_model.h5")
encoder = tf.keras.Model(inputs=autoencoder.input, outputs=autoencoder.layers[-7].output)  

train_embeddings = encoder.predict(x_train)
train_embeddings_flat = train_embeddings.reshape(len(train_embeddings), -1)

def find_similar_images(query_image, num_results=5):
    query_embedding = encoder.predict(query_image.reshape(1, 28, 28, 1)).reshape(1, -1)
    
    similarities = cosine_similarity(query_embedding, train_embeddings_flat)
    most_similar_indices = np.argsort(similarities[0])[::-1][:num_results]
    
    similar_images = x_train[most_similar_indices]
    similar_labels = y_train[most_similar_indices]
    
    return similar_images, similar_labels

def query_multiple_images(query_indices, num_results=5):
    plt.figure(figsize=(15, len(query_indices) * 3))
    
    for row, query_index in enumerate(query_indices):
        query_image = x_test[query_index]
        query_label = y_test[query_index]
        
        similar_images, similar_labels = find_similar_images(query_image, num_results=num_results)
        
        plt.subplot(len(query_indices), num_results + 1, row * (num_results + 1) + 1)
        plt.imshow(query_image.squeeze(), cmap='gray')
        plt.title(f"Query: {query_label}")
        plt.axis('off')
        
        for i, img in enumerate(similar_images):
            plt.subplot(len(query_indices), num_results + 1, row * (num_results + 1) + i + 2)
            plt.imshow(img.squeeze(), cmap='gray')
            plt.title(f"Label: {similar_labels[i]}")
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()


query_indices = np.random.choice(len(x_test), 5, replace=False)  
print("Query Indices:", query_indices)

query_multiple_images(query_indices)

test, lab_test = loading_5_per_class_lab('image_test',label)
query, lab_query= loading_5_per_class_lab('image_query', label_q)

#transform to tensor and normalize
test_tens= [transform(img) for img in test]

#transform to tensor and normalize
query_tens = [transform(img) for img in query]

with torch.no_grad():  # Disable gradient computation
    query_embeddings = [model(img.unsqueeze(0)) for img in query_tens]  # Compute embeddings for query images
    test_embeddings = [model(img.unsqueeze(0)) for img in test_tens] # Compute embeddings for test images

distances = distance.cdist(np.array(query_embeddings).squeeze(), np.array(test_embeddings).squeeze(), metric='euclidean')

#getting the indices of top 5 images for each query
top_k = 5
top_k_indices = np.argsort(distances, axis=1)[:, :top_k]

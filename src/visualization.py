query_index = 20 #vehicle to identify

# Retrieve the top 5 ranked test images based on the query
top_test_indices = top_k_indices[query_index]
top_test_images = [test[i] for i in top_test_indices]

# Plotting
plt.figure(figsize=(10, 6))

# Plot the query image
plt.subplot(1, top_k + 1, 1)
plt.imshow(query[query_index])
plt.title("Query Image")
plt.axis('off')

# Plot the top 5 ranked test images
for i in range(top_k):
    plt.subplot(1, top_k + 1, i + 2)
    plt.imshow(top_test_images[i])  
    plt.title(f"Rank {i + 1}")
    plt.axis('off')
plt.tight_layout()
plt.show()


batch_size = 100
num_epochs = 10

# Model and optimizer setup
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    # Shuffle the data manually
    combined = list(zip(list_anchor, list_pos, list_neg))
    anchors, positives, negatives = zip(*combined)
    
    # Loop through the dataset in batches
    for i in range(0, len(anchors), batch_size):
        batch_anch = torch.stack([transform(img) for img in anchors[i:i + batch_size]])
        batch_pos = torch.stack([transform(img) for img in positives[i:i + batch_size]])
        batch_neg = torch.stack([transform(img) for img in negatives[i:i + batch_size]])
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        output_anch = model(batch_anch)
        output_pos = model(batch_pos)
        output_neg = model(batch_neg)
        
        # Compute loss (triplet margin loss example)
        pos_dist = F.pairwise_distance(output_anch, output_pos, p=2)
        neg_dist = F.pairwise_distance(output_anch, output_neg, p=2)
        loss = F.triplet_margin_loss(output_anch, output_pos, output_neg, margin=1.0, p=2)
        
        # Backward and optimize
        loss.backward()
        optimizer.step()

#saving the parameters of the model
torch.save(model.state_dict(), 'embedding_model.pth')

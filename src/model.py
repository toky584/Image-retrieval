#download the pre trained model.
res_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
for param in res_model.parameters():
    param.requires_grad = False 

#creating the embedding model using the pretrained model
class EmbeddingModel(nn.Module):
    def __init__(self, res_model):
        super(EmbeddingModel, self).__init__()
        self.res_model = nn.Sequential(*list(res_model.children())[:-2])  
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))                 
        self.embedding = nn.Linear(res_model.fc.in_features, 416)          

    def forward(self, x):
        x = self.res_model(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)  
        x = self.embedding(x)    
        return x

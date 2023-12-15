import os
import torch
import argparse
import matplotlib.pyplot as plt
from torchvision import models
from sklearn.preprocessing import MultiLabelBinarizer
from load_data import get_data

def evaluate(model_path, test_loader, save_path='./saved_images'):
    loaded_model = torch.load(model_path, map_location=torch.device('cpu'))  # CPU로 로드
    loaded_model = loaded_model.eval()
    
    os.makedirs(save_path, exist_ok=True)
    
    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            images = sample['image']
            labels = sample['labels']
            
            outputs = loaded_model(images)
            predicted = (outputs > 0.5).float()  

            for idx in range(len(images)):
                image = images[idx].numpy()  
                true_label = labels[idx].numpy()  
                
                decoded_label = true_label
                
                predicted_label_encoded = predicted[idx].numpy()  
                
                predicted_label_decoded = predicted_label_encoded

                image_filename = f'batch_{i}_image_{idx}.jpeg'
                image_path = os.path.join(save_path, image_filename)
                plt.imshow(image.transpose(1, 2, 0))  
                plt.title(f'True Label: {decoded_label}, \n Predicted Label (Decoded): {predicted_label_decoded}', 
                          fontsize=10, multialignment='center')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(image_path)


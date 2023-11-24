import os
import torch
import matplotlib.pyplot as plt
from dataloaders.custom_dataset import mlb
from sklearn.preprocessing import MultiLabelBinarizer


def evaluate(model, test_loader, device, save_path='./saved_images'):
    model.eval()
    
    # Ensure the save_path directory exists
    os.makedirs(save_path, exist_ok=True)
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            # if i >= max_images:
            #     break  
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            predicted = (outputs > 0.5).float()  

            for idx in range(len(images)):
                # if idx >= max_images:
                #     break 
                image = images[idx].cpu().numpy()  
                true_label = labels[idx].cpu().numpy()  
                
                decoded_label = mlb.inverse_transform(true_label.reshape(1, -1))[0]
                
                predicted_label_encoded = predicted[idx].cpu().numpy()  
               
                predicted_label_decoded = mlb.inverse_transform(predicted_label_encoded.reshape(1, -1))[0]
                
                # Save the image to a file
                image_filename = f'batch_{i}_image_{idx}.jpeg'
                image_path = os.path.join(save_path, image_filename)
                plt.imshow(image.transpose(1, 2, 0))  
                plt.title(f'True Label: {decoded_label}, \n Predicted Label (Decoded): {predicted_label_decoded}', 
                          fontsize=10, multialignment='center')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(image_path)
                plt.show()
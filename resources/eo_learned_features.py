from tqdm import tqdm
import numpy as np
import pandas as pd

import torchvision.models as models
from torch import nn
# from torchsummary import summary # I don't think this is needed
from PIL import Image
from torchvision import transforms

class ResNet152(nn.Module):

    def __init__(self):
        super(ResNet152, self).__init__()
        
        # load the pretrained model
        self.model = models.resnet152(pretrained=True)

        # select till the last layer
        # Dropping output layer (the ImageNet classifier)
        self.model = nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, x):

      x = self.model(x)
      return x

def get_embedding(in_imgs):
    """Accepts train, dev, or test numpy matrix and
    returns numpy feature vector of shape (n_samples, 2048)
    where 2048 is the shape of the last hidden layer in pre-trained ResNet model"""

    out_feat = []

    model_conv_features = ResNet152()

    for i in range(in_imgs.shape[0]):

        # convert the grayscale to RGB images
        cur_rgb = np.stack([in_imgs[i,:,:], in_imgs[i,:,:], in_imgs[i,:,:]], axis=2)
        
        if np.max(cur_rgb)>1:
            cur_rgb = cur_rgb.astype(np.uint8)
        else:
            cur_rgb = (cur_rgb*255).astype(np.uint8)
        
        # preprocess the image to prepare it for input to CNN
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        out_im = preprocess(Image.fromarray(cur_rgb))
        out_feat.append(model_conv_features(out_im.unsqueeze(0).to('cpu')).squeeze().detach().numpy())

    return np.stack(out_feat, axis=0)

def get_eo_ndarray(df, rootdir="train3"):
    """Accepts train, dev, or test dataframes with image paths
    returns numpy feature vector of shape (n_samples, 32, 32)"""

    eo_images, eo_labels = [], []

    samples_per_train_class = df["label"].value_counts().sort_index().to_list()

    for i in tqdm(range(10)):
        for j in range(samples_per_train_class[i]):
            path_eo = rootdir + "/" + str(i) + "/" + df[df['label'] == i]['eo_img'].iloc[j]
            eo_img = Image.open(path_eo)
            eo_images.append(np.asarray(eo_img))
            eo_img.close()

            eo_labels.append(i)

    return np.stack(eo_images), np.array(eo_labels)

if __name__ == "__main__":
    # Example usage in main report notebook after imports:
    # Set up separate training, dev, and test set dataframes
    files = pd.read_csv('mavoc_partition_scheme.csv')
    files['eo_img'] = files['eo_img'].str[9:]

    train = files[files['partition'] == 'train']
    dev = files[files['partition'] == 'dev']
    test = files[files['partition'] == 'test']

    # Get the numpy array matrices of each partition
    train_features_arr, train_labels_arr = get_eo_ndarray(train)
    dev_features_arr, dev_labels_arr = get_eo_ndarray(dev)
    test_features_arr, test_labels_arr = get_eo_ndarray(test)

    print(f"Train array: {train_features_arr.shape} Train labels:{train_labels_arr.shape}")
    print(f"Dev array: {dev_features_arr.shape} Dev labels:{dev_labels_arr.shape}")
    print(f"Test array: {test_features_arr.shape} Test labels:{test_labels_arr.shape}")

    # Grab the numpy array matrices of each partition
    train_learned_features = get_embedding(train_features_arr) # pass this in as feature vector to classifier
    dev_learned_features = get_embedding(dev_features_arr)
    test_learned_features = get_embedding(test_features_arr)

    print(f"Train embedding: {train_learned_features.shape}")
    print(f"Dev embedding: {dev_learned_features.shape}")
    print(f"Test embedding: {test_learned_features.shape}")
    
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from glob import glob
import seaborn as sn

from torch import nn
import torchvision.models as models
from torchvision import transforms
from torchsummary import summary

#from skimage.exposure import histogram
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage import data, exposure, transform, color

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Path to Dataset
root_path = r"C:\Users\alexs\Downloads\Intel Training Dataset\Intel Training Dataset"

# split into subfolders based on class label
subfolders = sorted(glob(root_path + '\\*'))
label_names = [p.split('/')[-1] for p in subfolders]
label_basenames = [p.split('\\')[-1] for p in subfolders]

# load nn model
resnet50 = models.resnet50(pretrained=True)

# get layers
def slice_model(original_model, from_layer=None, to_layer=None):
  return nn.Sequential(*list(original_model.children())[from_layer:to_layer])

model_conv_features = slice_model(resnet50, to_layer=-1).to('cpu')
summary(model_conv_features, input_size=(3, 224, 224))

# preprocess
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def retype_image(in_img):
  if np.max(in_img) > 1:
    in_img = in_img.astype(np.uint8)
  else:
    in_img = (in_img * 255.0).astype(np.uint8)
  return in_img

# put the network in evaluation mode
resnet50.eval()

fname = sorted(glob(subfolders[0] + '/*.jpg'))[0]
test_img = plt.imread(fname)

total_time = 0
# create a list to organize labels, filenames, and feature vectors
start_time = time.time()

data = []
num_per_class=200

for i, (label, subfolder) in enumerate(zip(label_names, subfolders)):
    # get list of file paths for each subfolder
    file_paths = sorted(glob(subfolder + '/*.jpg'))
    for f in file_paths[:num_per_class]:
        #for flip in [False, True]:

            # read image
            img = np.array(Image.open(f))
            
            proc_img = preprocess(Image.fromarray(retype_image(img)))
            feat = model_conv_features(proc_img.unsqueeze(0).to('cpu')).squeeze().detach().numpy()

            #if flip:
                #img = np.flipud(img)
            img_shrunk = transform.resize(img, (10, 10))
            img_flat = img.flatten()
            r_channel=img_shrunk[:,:,0].flatten()
            g_channel=img_shrunk[:,:,1].flatten()
            b_channel=img_shrunk[:,:,2].flatten()

            gray_img = rgb2gray(img)
            fname = f.split('/')[-1].split('_')[-1]
            # convert to luminance histogram (feature vector)
            img_hist, _ = np.histogram(gray_img, bins=50, 
                                    range=(0,1), 
                                    density=True)
            
            
            # convert to HOG
            fd, img_hog = hog(gray_img, orientations=8, pixels_per_cell=(16, 16),
                            cells_per_block=(1, 1), visualize=True, block_norm = 'L2')
            
            #normalizing: scaling bin values to be between 0 and 1 by dividing each bin by sum of all bins
            
            r_channel = r_channel / np.sum(r_channel)
            g_channel = g_channel / np.sum(g_channel)
            b_channel = b_channel / np.sum(b_channel)
            img_hist = img_hist / np.sum(img_hist)
            fd=fd/np.sum(fd)

            # append to data list with labels
            data.append({'labelname':label, 
                        'filename':fname, 
                        'labelnum':i, 
                        'lumhist':img_hist,
                        'hog_rgb': np.concatenate((fd, r_channel, g_channel, b_channel)),
                        'nn_features': feat
                        }) 
# convert to dataframe for storage
# can also export to a file here
df = pd.DataFrame(data=data)
end_time = time.time()
total_time += end_time-start_time
print("Feature Vector Time: ", end_time-start_time)

# re-load data
label_array = df['labelnum'] # vector

feature_matrix1 = np.vstack(df['lumhist'])
feature_matrix2 = np.vstack(df['hog_rgb'])
feature_matrix3 = np.vstack(df['nn_features'])

feature_matrix = np.hstack([feature_matrix1, feature_matrix2, feature_matrix3])

X_temp, X_test, y_temp, y_test = train_test_split(
    feature_matrix,
    label_array,
    test_size=0.4,
    stratify=label_array,
    random_state=0,
)

#split temp set into validation and test sets
X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    stratify=y_temp,
    random_state=0,
)


start_time = time.time()
# train a simple classifier
clf = make_pipeline(StandardScaler(), SVC(C=10, gamma=0.001, kernel='rbf'))
clf.fit(X_train, y_train)

# report overall test accuracy
end_time = time.time()
total_time += end_time-start_time
print('Total SVC Train Accuracy: {}'.format(clf.score(X_train, y_train)))
print('Total SVC Test Accuracy: {}'.format(clf.score(X_test, y_test)))
print('Total SVC Validation Accuracy: {}'.format(clf.score(X_val, y_val)))
print("Total SVC training time : ", end_time-start_time)

#train random forest

start_time = time.time()
rf_clf = RandomForestClassifier(n_estimators=100, random_state=69420)
rf_clf.fit(X_train, y_train)

end_time = time.time()
total_time += end_time-start_time
print('Random Forest Train Accuracy: {}'.format(rf_clf.score(X_train, y_train)))
print('Random Forest Test Accuracy: {}'.format(rf_clf.score(X_test, y_test)))
print('Random Forest Validation Accuracy: {}'.format(rf_clf.score(X_val, y_val)))
print("Total Random Forest training time: ", end_time-start_time)

start_time = time.time()

y_pred = clf.predict(X_test)

# Plot the results as a confusion matrix
C = confusion_matrix(y_test, y_pred)

heatmap = sn.heatmap(C, annot=True, cmap='inferno', fmt=".0f", xticklabels=label_basenames, yticklabels=label_basenames)
end_time = time.time()
total_time += end_time-start_time
print("Test/Predict time: ", end_time-start_time)


# Report random forest
start_time = time.time()
y_pred = rf_clf.predict(X_test)


C = confusion_matrix(y_test, y_pred)

heatmap = sn.heatmap(C, annot=True, cmap='inferno', fmt=".0f", xticklabels=label_basenames, yticklabels=label_basenames)
end_time = time.time()
total_time += end_time-start_time
print("Test time: ", end_time-start_time)

print("Total compute time: ", total_time)

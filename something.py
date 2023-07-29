import os
import matplotlib.pyplot as plt
import matplotlib.image as IM


data_dir = "data"
imgs = []

for image_class in os.listdir(data_dir): 
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        imgs.append(image_path)

        break

fig, ax = plt.subplots(ncols=5, figsize=(20,20))
for idx, img in enumerate(imgs):
    ax[idx].imshow(IM.imread(img))
    ax[idx].title.set_text(os.listdir(data_dir)[idx])
plt.show()
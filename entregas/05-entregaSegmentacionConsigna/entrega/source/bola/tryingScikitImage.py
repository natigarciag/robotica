import numpy as np
import matplotlib.pyplot as plt
import skimage.data as data
import skimage.segmentation as seg
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color

def image_show(image, nrows=1, ncols=1, cmap='gray'):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
    ax.imshow(image, cmap='gray')
    plt.show()
    ax.axis('off')
    return fig, ax


text = data.page()
image_show(text)

fig, ax = plt.subplots(1, 1)
ax.hist(text.ravel(), bins=32, range=[0, 256])
ax.set_xlim(0, 256)
plt.show()

image_slic = seg.slic(text,n_segments=2)
image_show(color.label2rgb(image_slic, text, kind='avg'))

# image = data.binary_blobs()
# plt.imshow(image)
# plt.show()
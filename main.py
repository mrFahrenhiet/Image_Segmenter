#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

img = cv2.imread('./elephant.jpg')
print(img.shape)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
plt.axis(False)
plt.show()

img = img.reshape((-1, 3))

print(img.shape)

k = 4
kms = KMeans(n_clusters=4)

kms.fit(img)

centers = kms.cluster_centers_

centers = np.array((centers), dtype='uint8')
print(centers)

i = 1
plt.figure(0, figsize=(8, 2))
color = []
for each_color in centers:
    plt.subplot(1, 4, i)
    plt.axis(False)
    i += 1
    color.append(each_color)
    a = np.zeros((100, 100, 3), dtype='uint8')
    a[:, :, :] = each_color
    plt.imshow(a)
plt.show()

new_img = np.zeros((330 * 500, 3), dtype='uint8')
print(color)


for i in range(new_img.shape[0]):
    new_img[i] = color[kms.labels_[i]]
new_img = new_img.reshape((330, 500, 3))
plt.imshow(new_img)
plt.show()

im2 = np.zeros((300 * 550, 3), dtype='uint8')

for i in range(im2.shape[0]):
    if kms.labels_[i] == 0:
        im2[i] = color[kms.labels_[i]]
    else:
        im2[i] = np.array([255, 255, 255], dtype='uint8')

im2 = im2.reshape((330, 500, 3))
plt.imshow(im2)
plt.show()


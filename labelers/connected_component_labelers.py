from abc import ABC, abstractmethod
from colordict import ColorDict
from PIL import Image
import numpy as np



class ConnectedComponentLabeler(ABC):
    labeler_type = None

    @classmethod
    def get_labeler(cls, labeler_type: str):
        return next(x for x in cls.__subclasses__() if x.labeler_type == labeler_type)()

    @abstractmethod
    def label_components(self, B):
        pass


class RecursiveConnectedComponentLabeler(ConnectedComponentLabeler):
    labeler_type = "recursive"

    def find_components(self, label_img, label):
        max_rows, max_cols = label_img.shape
        for i in range(max_rows):
            for j in range(max_cols):
                if label_img[i, j] == -1:
                    label = label + 1
                    self.search(label_img, label, i, j)

    def search(self, label_img, label, i, j):
        label_img[i, j] = label
        neighborhood = label_img[i - 1:i + 2, j - 1:j + 2]
        for n in range(neighborhood.shape[0]):
            for m in range(neighborhood.shape[1]):
                if neighborhood[n, m] == -1:
                    self.search(label_img, label, i + n - 1, j + m - 1)

    def label_components(self, binary_img):
        label_img = -binary_img
        label = 0
        self.find_components(label_img, label)
        return label_img


def prior_neighbors(img, i, j):
    neighbors = []
    if i == 0:
        A = img[i, j - 1:j]
    elif j == 0:
        A = img[i - 1:i, j]
    else:
        A = img[i - 1:i + 1, j - 1:j + 2].flatten()[:-2]
    for i in range(A.shape[0]):
        if A[i] == 1:
            neighbors.append(i)
    return neighbors


def get_labels(label_img, i, j, indices):
    if i == 0:
        labels = label_img[i, j - 1:j]
    elif j == 0:
        labels = label_img[i - 1:i + 1, j:j + 2].flatten()[:-1]
    else:
        labels = label_img[i - 1:i + 1, j - 1:j + 2].flatten()[:-2]
    neighboring_labels = []
    for index in indices:
        neighboring_labels.append(labels[index])
        
    return neighboring_labels


class UnionFindConnectedComponentLabeler(ConnectedComponentLabeler):
    labeler_type = "union"

    def __init__(self):
        self.parent = []
        for _ in range(0, 100000000):
            self.parent.append(0)
        self.num_labels = 0

    def union(self, X, Y):
        j = int(X)
        k = int(Y)
 
        while self.parent[j] != 0:
            j = self.parent[j]
        while self.parent[k] != 0:
            k = self.parent[k]
        if k != j:
            self.parent[k] = j

    def find(self, X):
        j = int(X)
        while self.parent[j] != 0:
            j = self.parent[j]
        return j
        
    def label_components(self, binary_image):
        list_label = []
        label = 10
        label_image = np.zeros(binary_image.shape)
        max_rows, max_cols = binary_image.shape
        for i in range(max_rows):
            for j in range(max_cols):
                if binary_image[i, j] == 1:
                    A = prior_neighbors(binary_image, i, j)
                    if len(A) == 0:
                        m = label;
                        label = label + 1
                    else:
                        m = int(np.amin(get_labels(label_image, i, j, A)))
                    
                    #print("label: ",label, ",m: ",m, ", A: ",A)
                    label_image[i, j] = m
                    
                    labels = get_labels(label_image, i, j, A)
                    for l in labels:
                        if l != m:
                            self.union(m, l)

        # create output
        r, c = label_image.shape
        out_image = Image.new("RGB", (r, c),0)
        pix = out_image.load()
        colors = list(ColorDict().values())
            
        for i in range(max_rows):
            for j in range(max_cols):
                if binary_image[i, j] == 1:
                    label_image[i, j] = self.find(label_image[i, j])
                    if label_image[i, j] not in list_label:
                        list_label.append(label_image[i, j])
                    
                    # add value color for out image
                    value = int(label_image[i, j])
                    if list_label.index(value) <= len(colors):
                        color = colors[list_label.index(value)]
                        pix[i,j] = (int(color[0]),int(color[1]),int(color[2]),int(color[3]))
                    else:
                        out_image = []
    
        return label_image,list_label,out_image

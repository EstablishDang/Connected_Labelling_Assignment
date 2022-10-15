import math
import argparse
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image,ImageOps
from colordict import ColorDict
import random
from labelers.connected_component_labelers import ConnectedComponentLabeler

list_color =[(255,0,0),(255,255,0),(0,255,0),(0,255,255),(0,0,255),(255,0,255),(160,0,0),(0,160,0),(0,0,160),(160,160,0),(0,160,160),(160,0,160),(50,50,100),(255,50,50)]
def image_to_binary(image):
    img = image.convert('L')
    arr = np.asarray(img)
    # bin_img = np.where(arr<128, 1, 0).astype(int)
    bin_img = (arr > 128).astype(int)
    #bin_img = arr.astype(float) / 255
    return bin_img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--labeler", help="Labeler type", type=str, default="union")
    parser.add_argument('images', nargs='*')
    args = parser.parse_args()

    if not args.images:
        images = ["test_data/example_img_1.txt", "test_data/example_img_2.txt", "test_data/example_img_3.txt"]
    else:
        images = args.images

    imgs = []
    
    image = Image.open(images[0])
    image = ImageOps.mirror(image)
    binary_image = image_to_binary(image)
    #np.savetxt('matrix_text.txt', binary_image, delimiter=',', fmt='%i')
    
    # for arg in images:
    #     text_file = open(arg, "r")
    #     lines = text_file.read().split(',')
    #     vals = [int(line) for line in lines]
    imgs.append(binary_image)

    for img in imgs:
        labeler = ConnectedComponentLabeler.get_labeler(args.labeler)
        
        #s = int(math.sqrt(img.shape[0]))
        #img = np.reshape(img, (s, s))
        #print(img.size)
        #img = np.reshape(img, (s, s))

        labeled_img,list_label,out_image = labeler.label_components(img)
        #print(list_label)
        
        #colors = list(ColorDict().values())
        #print(colors[0][0])
        
        #np.savetxt('img1.txt', labeled_img, delimiter=',', fmt='%i')

        # im = Image.fromarray('1', labeled_img)
        # im.save("test.png", "PNG")
        
        # if len(list_label) <= len(list_color):
        #r, c = labeled_img.shape
        #im = Image.new("RGB", (r, c),0)
        #pix = im.load()
        
        # for x in range(r):
        #     for y in range(c):
        #         if int(labeled_img[x,c-y-1]) == 0:
        #             color = (0,0,0,0)
        #         else:
        #             value = int(labeled_img[x,c-y-1])
        #             color = colors[list_label.index(value)]

        #         pix[x,y] = (int(color[0]),int(color[1]),int(color[2]),int(color[3]))
        if out_image != []:
            out_image.save("figures/out_put.png", "PNG")
        else:
            fig = plt.figure()
            ax = fig.add_subplot()
            plt.imshow(labeled_img)
            plt.show()
        
        # else:
        #     fig = plt.figure()
            # ax = fig.add_subplot()
            # plt.imshow(labeled_img)
            # plt.show()
        # ax = fig.add_subplot(121)
        # plt.imshow(img, cmap='gray')
        # plt.axis('off')
        # plt.title("Original Image")

            
        # plt.axis('off')
        # plt.title("Labeled Image")

        # for (j, i), label in np.ndenumerate(labeled_img):
        #     ax.text(i, j, int(label), ha='center', va='center')

            
        del labeler


if __name__ == "__main__":
    main()
    exit()

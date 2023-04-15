import numpy as np
from PIL import Image
from sympy import Matrix, pprint


# da_image = 'Salad_8x8.jpg'
# da_image = 'Salad_16x16.jpg'
# da_image = 'Salad_32x32.jpg'
# da_image = 'Salad_64x64.jpg'
# da_image = 'Salad_128x128.jpg'
da_image = 'Salad_256x256.jpg'
# da_image = 'Salad_512x512.jpg'
# da_image = 'Salad_1024x1024.jpg'
# da_image = 'Salad_2048x2048.jpg'
# da_image = 'Salad_4096x4096.jpg'


def main():

    filtr_size = 4 # since all filters are NxN, write just N

    filtr = average_filter(filtr_size)
    # filtr = []
    # filtr = []

    image = Image.open('img/'+da_image) # loading the image
    image.show()
    # output_image = Image.new(mode="RGB", size=(8, 8))
    # output_image.putpixel((1,1),(255,255,255))
    for i in range(0,1):
        image = convolve(image,filtr)
    
    image.show()


def convolve(image, filtr): # the convolution of some image with some filter
    output_image = image.copy()
    image_data = np.asarray(image)
    for i in range(0,image.size[0]-1): # looping through rows
        for j in range(0,image.size[1]-1): # looping through columns
            pixel_value = []
            for k in range(0,3): # looping through bands (RGB colors)
                image_segment = []
                for p in range(round(-len(filtr)/2)-1,round(len(filtr)/2)): # moving through the filter's rows
                    temp_segment = []
                    for q in range(round(-len(filtr)/2)-1,round(len(filtr)/2)): # moving through the filter's columns
                        temp_segment.append(image_data[j+p][i+q][k])
                    image_segment.append(temp_segment)
                # print(image_segment)
                pixel_value.append(total_sum_from_filter(filtr, image_segment))
            pixel_value = tuple(pixel_value)
            output_image.putpixel((i,j), pixel_value)
    return output_image

def assemble_image_segement(image, size):
    image_segment = []
    for p in range(round(-size/2),round(size/2)):
        temp_list_item = []
            

def total_sum_from_filter(filtr, image_segment): # image segment is going to be the same size as the filter
    sum = 0
    for i in range(0,len(filtr)):
        for j in range(0,len(filtr[0])):
            sum = sum + filtr[i][j]*image_segment[i][j]
    return round(sum)

def average_filter(window_size): # N, not NxN datatype
    final_filtr = []
    for i in range(0,window_size):
        row_of_filtr = []
        for j in range(0,window_size):
            row_of_filtr.append(1/pow(window_size,2))
        final_filtr.append(row_of_filtr)
    return final_filtr


main()

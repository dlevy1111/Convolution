import numpy as np
from PIL import Image, ImageOps
from sympy import Matrix, pprint

import sys # for fun 'in action' meter

##### Issues that need to be fixed/implemented:
# Arbitrary filter size: ✓ ✓
# Arbitrary filter: ✓ (sort of, implementing as necessary)
# Better edge cases: ✓
# Arbitrary stride: sorta??
# progress meter ✓

#-----------------------------------------------------------------------------#

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

num_bands = 1 # 1 for grayscale, 3 for RGB

#-----------------------------------------------------------------------------#

def main(da_image):
    image = Image.open(da_image) # loading the image
    if num_bands == 1:
        image = ImageOps.grayscale(image)

    #-----------------------------------------------------------------------------#

    filtr_size = 3 # since all filters are NxN, write just N. things might break for #s greater than 10 or so, i haven't tried very many
    
    blurring_factor = 1 # larger will make the picture dimmer, smaller will make the picture brighter

    # filtr = average_filter(filtr_size)
    # filtr = outline_filter() # outline edge detection filter
    # filtr = sobel_filter()
    # filtr = prewitt_filter()
    filtr = gaussian_filter(filtr_size, blurring_factor)

    dilation_size = 1 # stride size must be greater than 1, so when adjusting it, make sure it's >116
    
    edges_mode = 'black_edges' # deal with edge cases by padding with black pixels?
    # edges_mode = 'pretty_edges' # deal with edge cases by padding with extensions of colored pixels?
    # edges_mode = 'no_edges' # deal with edge cases by having the filter return 0 for any edges?
    # edges_mode = 'wrap_edges' # deal with edge cases by having the filter wrap the image

    times_convolved = 1 # how many times would you like to run convolution? (1 is best for edge stuff)

    #-----------------------------------------------------------------------------#

    color_scheme = ""
    if num_bands == 1:
        color_scheme = "L"
    else:
        color_scheme = "RGB"

    # image.show()
    # print(image.size[0], image.size[1])

    if edges_mode == 'black_edges':
        image = add_black_boarder_to_image(image, round(filtr_size/2)) # add a black boarder as a method for dealing with edge cases
    if edges_mode == 'pretty_edges':
        image = add_matched_boarder_to_image(image, round(filtr_size/2)) # add colored pixels to the boarder such that the pixels are colored like nearest neighbors
    if edges_mode == 'no_edges':
        a = 'very sadly, we do nothing here. go look at convolve() though'
    if edges_mode == "wrap_edges":
        image = add_wrapped_edges_to_image(image, round(filtr_size/2))
    
    # image.show()
    if filtr_size > len(filtr):
        new_filter = np.zeros((filtr_size, filtr_size))
        for i in range(0,len(filtr)):
            for j in range(0,len(filtr)):
                new_filter[i][j] = filtr[i][j]
        filtr = new_filter.tolist()
    
    # print(filtr)

    print("Progress", end="") # makes sense in the context of the output
    for i in range(0,times_convolved):
        print(f"\n{i+1}/{times_convolved}:")
        image = convolve(image, filtr, edges_mode, dilation_size, color_scheme)
    print("\nDone")

    # image.show()

    # cropping the image back to the original size
    if edges_mode == 'black_edges' or edges_mode == 'pretty_edges' or edges_mode == "wrap_edges":
        if filtr_size == 3: # special case 3
            image = image.crop(((filtr_size)//2+1, (filtr_size)//2+1, image.size[0]-(filtr_size)//2-1, image.size[0]-(filtr_size)//2-1))
        elif filtr_size == 7:
            image = image.crop(((filtr_size+1)//2, (filtr_size+1)//2, image.size[0]-(filtr_size+1)//2, image.size[0]-(filtr_size+1)//2))
        elif filtr_size%2 == 1: # if the filter is an odd size
            image = image.crop(((filtr_size+0.5)//2, (filtr_size+0.5)//2, image.size[0]-(filtr_size-0.5)//2, image.size[0]-(filtr_size-0.5)//2))
        else: # the filter is an even size
            image = image.crop(((filtr_size)//2, (filtr_size)//2, image.size[0]-(filtr_size)//2, image.size[0]-(filtr_size)//2))
    image.show()
    # print(image.size[0], image.size[1])
    image.save(f"jayhawk_out/output_{da_image[15:]}")

# does not shrink the images
def convolve(image, filtr, mode, dilation_size, color_mode): # the convolution of some image with some filter
    output_image = image.copy()
    image_data = np.asarray(image)
    offset = 0
    if mode == 'no_edges':
        offset = len(filtr)//2+1
    for i in range(len(filtr)//2-offset, image.size[0]-len(filtr)//2+offset, dilation_size): # looping through rows
        for j in range(len(filtr)//2-offset, image.size[1]-len(filtr)//2+offset, dilation_size): # looping through columns
            
            # fun progress bar so you know how long until your program finishes (roughly)
            sys.stdout.write('\r')
            sys.stdout.write("[%-50s] %d%%" % ('='*int(i*50/((image.size[0]-len(filtr)//2+offset)^2)), int((i)*100/((image.size[0]-len(filtr)//2+offset)^2))))
            sys.stdout.flush()
            
            pixel_value = []
            for k in range(0,num_bands): # looping through bands (RGB colors, grayscale)
                image_segment = []
                for p in range(round(-len(filtr)/2)+1,round(len(filtr)/2)): # moving through the filter's rows
                    temp_segment = []
                    for q in range(round(-len(filtr)/2)+1,round(len(filtr)/2)): # moving through the filter's columns
                        if color_mode == "RGB":
                            temp_segment.append(image_data[j+p][i+q][k]) # multi-band case
                        elif color_mode == "L":
                            temp_segment.append(image_data[j+p][i+q]) # RGB case, we remove k from this since when k=1, the image_data is going to be an integer
                        else:
                            temp_segment.append(0) # for the no-edges case
                    image_segment.append(temp_segment)
                pixel_value.append(total_sum_from_filter(filtr, image_segment))
            pixel_value = tuple(pixel_value)
            output_image.putpixel((i,j), pixel_value)
    return output_image

def add_black_boarder_to_image(image, size_of_boarder): # for edge cases; size_of_boarder should be an integer, as the total extension is measured from one side
    new_image = ImageOps.expand(image, size_of_boarder)
    return new_image

def add_matched_boarder_to_image(image, size_of_boarder): # for edge cases; adds a boarder that is the same color as the nearest pixel
    new_image = ImageOps.expand(image, size_of_boarder) # first we make the image larger
    new_image_data = np.asarray(new_image)
    # fixing vertically and horizontally (top and bottom sections, and right and left sections)
    for j in range(0, image.size[0]+size_of_boarder): # j loops through columns
        for m in range(0, size_of_boarder):
            # fixing vertically
            if type(new_image_data[size_of_boarder][j]) == np.ndarray: # if we're working with more than 1 band
                new_image.putpixel((j,m), tuple(new_image_data[size_of_boarder][j])) # fixing from the top
                new_image.putpixel((j,image.size[0]-m+2*size_of_boarder-1), tuple(new_image_data[image.size[0]+size_of_boarder-1][j])) # fixing from the bottom
                # fixing horizontally
                new_image.putpixel((m,j), tuple(new_image_data[j][size_of_boarder])) # fixing from the left
                new_image.putpixel((image.size[0]+m+size_of_boarder,j), tuple(new_image_data[j][image.size[0]+size_of_boarder-1])) # fixing from the right
            
            if type(new_image_data[size_of_boarder][j]) == np.uint8: # if we're working with only 1 band
                new_image.putpixel((j,m), int(new_image_data[size_of_boarder][j])) # fixing from the top
                new_image.putpixel((j,image.size[0]-m+2*size_of_boarder-1), int(new_image_data[image.size[0]+size_of_boarder-1][j])) # fixing from the bottom
                # fixing horizontally
                new_image.putpixel((m,j), int(new_image_data[j][size_of_boarder])) # fixing from the left
                new_image.putpixel((image.size[0]+m+size_of_boarder,j), int(new_image_data[j][image.size[0]+size_of_boarder-1])) # fixing from the right

    # fixing corners
    for i in range(0,size_of_boarder):
        for j in range(0,size_of_boarder):
            if type(new_image_data[size_of_boarder][j]) == np.ndarray: # if we're working with more than 1
                new_image.putpixel((i,j), tuple(new_image_data[size_of_boarder+1][size_of_boarder+1])) # top left
                new_image.putpixel((new_image.size[0]-i-1,j), tuple(new_image_data[size_of_boarder+1][image.size[0]+size_of_boarder-2])) # top right
                new_image.putpixel((new_image.size[0]-i-1, new_image.size[0]-j-1), tuple(new_image_data[new_image.size[0]-size_of_boarder-1][new_image.size[0]-size_of_boarder-1])) # bottom right
                new_image.putpixel((i, new_image.size[0]-j-1), tuple(new_image_data[image.size[0]+size_of_boarder-2][size_of_boarder+1])) # bottom left
            
            if type(new_image_data[size_of_boarder][j]) == np.uint8: # if we're working with only 1 band
                new_image.putpixel((i,j), int(new_image_data[size_of_boarder+1][size_of_boarder+1])) # top left
                new_image.putpixel((new_image.size[0]-i-1,j), int(new_image_data[size_of_boarder+1][image.size[0]+size_of_boarder-2])) # top right
                new_image.putpixel((new_image.size[0]-i-1, new_image.size[0]-j-1), int(new_image_data[new_image.size[0]-size_of_boarder-1][new_image.size[0]-size_of_boarder-1])) # bottom right
                new_image.putpixel((i, new_image.size[0]-j-1), int(new_image_data[image.size[0]+size_of_boarder-2][size_of_boarder+1])) # bottom left
    return new_image

def add_wrapped_edges_to_image(image, size_of_boarder):
    new_image = ImageOps.expand(image, size_of_boarder)
    new_image_data = np.asarray(new_image)
    # fixing vertically and horizontally (top and bottom sections, and right and left sections)
    for j in range(0, image.size[0]+size_of_boarder): # j loops through columns
        for m in range(0, size_of_boarder):
            if type(new_image_data[size_of_boarder][j]) == np.ndarray: # if we're working with more than 1
                # fixing from the top
                new_image.putpixel((j,m), tuple(new_image_data[m-image.size[0]-size_of_boarder][j])) # fixing from the top
                new_image.putpixel((j,image.size[0]+size_of_boarder+m), tuple(new_image_data[size_of_boarder+m][j])) # fixing from the bottom
                # fixing from the sides
                new_image.putpixel((m,j), tuple(new_image_data[j][m-image.size[0]-size_of_boarder])) # fixing from the left
                new_image.putpixel((image.size[0]+size_of_boarder+m,j), tuple(new_image_data[j][size_of_boarder+m])) # fixing from the right
            if type(new_image_data[size_of_boarder][j]) == np.uint8: # if we're working with more than 1
                # fixing from the top
                new_image.putpixel((j,m), int(new_image_data[m-image.size[0]-size_of_boarder][j]))
                new_image.putpixel((j,image.size[0]+size_of_boarder+m), int(new_image_data[size_of_boarder+m][j]))
                # fixing from the sides
                new_image.putpixel((m,j), int(new_image_data[j][image.size[0]-(size_of_boarder-m)]))
                new_image.putpixel((image.size[0]+size_of_boarder+m,j), int(new_image_data[j][size_of_boarder+m]))
    
    # fixing corners
    for i in range(0, size_of_boarder):
        for j in range(0, size_of_boarder):
            if type(new_image_data[size_of_boarder][j]) == np.ndarray: # if we're working with more than 1 band
                new_image.putpixel((i,j), tuple(new_image_data[new_image.size[0]-size_of_boarder-1][new_image.size[0]-size_of_boarder-1])) # bottom right in the top left
                new_image.putpixel((new_image.size[0]-i-1,j), tuple(new_image_data[image.size[0]+size_of_boarder-2][size_of_boarder+1])) # bottom left in the top right
                new_image.putpixel((new_image.size[0]-i-1, new_image.size[0]-j-1), tuple(new_image_data[size_of_boarder+1][size_of_boarder+1])) # top left in the bottom right
                new_image.putpixel((i, new_image.size[0]-j-1), tuple(new_image_data[size_of_boarder+1][image.size[0]+size_of_boarder-2])) # top right in the bottom left
            
            if type(new_image_data[size_of_boarder][j]) == np.uint8: # if we're working with only 1 band
                new_image.putpixel((i,j), int(new_image_data[new_image.size[0]-size_of_boarder-1][new_image.size[0]-size_of_boarder-1])) # bottom right in the top left
                new_image.putpixel((new_image.size[0]-i-1,j), int(new_image_data[image.size[0]+size_of_boarder-2][size_of_boarder+1])) # bottom left in the top right
                new_image.putpixel((new_image.size[0]-i-1, new_image.size[0]-j-1), int(new_image_data[size_of_boarder+1][size_of_boarder+1])) # top left in the bottom right
                new_image.putpixel((i, new_image.size[0]-j-1), int(new_image_data[size_of_boarder+1][image.size[0]+size_of_boarder-2])) # top right in the bottom left
    return new_image
            
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

def outline_filter():
    return [[-1, -1, -1],
            [-1, 8 , -1],
            [-1, -1, -1]]

def sobel_filter():
    return [[-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]]

def prewitt_filter():
    return [[-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]]

def gaussian_filter(filtr_size, blurring_factor): # complete function for calculating the gaussian filter
    output_filtr = []
    for i in range((-filtr_size)//2+1, filtr_size//2+1): # going through rows
        filtr_row = []
        for j in range((-filtr_size)//2+1, filtr_size//2+1): # going through columns 
            filtr_row.append(gauss_func(i, j, blurring_factor)) # calling the gauss function below to calculate the value at each position
        output_filtr.append(filtr_row)
    return output_filtr

def gauss_func(x, y, b): # b = blurring factor, or sigma (std dev)
    return (1/(2*np.pi*(b**2)))*np.exp(-(x**2+y**2)/(2*(b**2))) # classic gaussian function

if __name__ == '__main__':
    main(image)

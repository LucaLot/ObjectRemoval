import argparse
import PySimpleGUI as sg
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')


class ImageDisplay:
    def __init__(self, image, width, height, filename) -> None:
        self.image = image
        self.height = height
        self.width = width
        self.filename = filename
        self.halfWidth = 0
        self.before_loc = (0, height)
        self.after_loc = (width,height*2)
        self.layout = self.create_layout()
        self.window = None
        self.contours, self.mask = self.initMask()
   
    def create_layout(self):
        layout = [[sg.pin(sg.Graph(
            canvas_size=(self.width, self.height),
            graph_bottom_left=(0, 0),
            graph_top_right=(self.width, self.height),
            key='-IMAGE-',
            background_color='white',
            change_submits=True,
            enable_events=True,
            drag_submits=True))]]
        return layout
    
    def create_window(self):
        image_data = np_im_to_data(self.image)
        window = sg.Window('Display Image', self.layout, finalize=True)    
        window['-IMAGE-'].draw_image(data=image_data, location=self.before_loc)
        return window
    
    def initMask(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (7,7), cv2.BORDER_DEFAULT)
        CannyAccThresh, matrix = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU);
        CannyThresh = 0.1 * CannyAccThresh;
        canny = cv2.Canny(blur,CannyThresh,CannyAccThresh);
        contours, hier = cv2.findContours(canny, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        mask = np.zeros(blur.shape, dtype=np.uint8)
        for i in range(len(contours)):
            mask = cv2.fillConvexPoly(mask, contours[i], i)
        mask.tofile('save.csv', sep=',', format='%s')
        return contours, mask
    
    def update_GUI(self, new_image):
        self.image = new_image
        self.layout = self.create_layout()
        self.window.close()
        self.window = self.create_window()

    def remove(self, e):
        """
        altered_image = np.zeros(self.image.shape)
        value = self.mask[e.y, e.x]
        for x in range(len(self.image[1])):
            for y in range(len(self.image[0])):
                altered_image[y,x] = [0,0,0] if self.mask[y,x] == value else self.image[y,x]"""
        tempMask = np.array(cv2.fillConvexPoly(np.zeros(shape=(self.height, self.width)), self.contours[self.mask[e.y, e.x]], 255), dtype=np.uint8)
        print(tempMask.dtype)
        altered_image = cv2.inpaint(src=self.image, inpaintMask=tempMask, inpaintRadius=5, flags=cv2.INPAINT_NS)
        self.update_GUI(altered_image)
        

def np_im_to_data(im):
    array = np.array(im, dtype=np.uint8)
    im = Image.fromarray(array)
    with BytesIO() as output:
        im.save(output, format='PNG')
        data = output.getvalue()
    return data
#Helper function to update the image of the GUI

def load_image(image_file):
    print(f'Loading {image_file} ... ', end='')
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f'{image.shape}')
    
    ratio = image.shape[1]/640
    height = int(image.shape[0]/ratio)
    width = int(image.shape[1]/ratio)


    print(f'Resizing the image to',height,'x',width,'...', end='')
    image = cv2.resize(image, (width,height), interpolation=cv2.INTER_LINEAR)
    print(f'{image.shape}')
    im = ImageDisplay(image, width, height, image_file)
    im.window = im.create_window()
    return im
#If no image is provided on load
def load_blank_image(image):
    height = 360
    width = 640
    image_file = 'test.jpg'
    im = ImageDisplay(image, width, height, image_file)
    im.window = sg.Window('Display Image', im.layout, finalize=True)    
    return im
#If an image is provided, or a new one is loaded
def display_image(image):
    # Event loop
    if image is not None:
        im = load_image(image)
    else:
        im = load_blank_image(image)
    
    while True:
        im.window.bind("<Motion>", "Motion")
        event, values = im.window.read()
        
        if event == '-IMAGE-+UP':
            e = im.window.user_bind_event
            im.remove(e)
        if event == sg.WINDOW_CLOSED or event == 'Quit':
            break

def main():
    parser = argparse.ArgumentParser(description='A simple image viewer.')

    parser.add_argument('file', action='store', nargs='?', const=0, default=None, help='Image file.')
    args = parser.parse_args()
    display_image(args.file)
    
if __name__ == '__main__':
    main()
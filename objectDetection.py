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
        self.originalImage = image
        self.image = image
        self.altered_image  = image
        self.height = height
        self.width = width
        self.filename = filename
        self.focus = True;
        self.only_threshold = False
        self.before_loc = (0, height)
        self.after_loc = (width,height*2)
        self.layout = self.create_layout()
        self.window = None
        self.edges, self.contours, self.mask = self.initMask()
   
    def create_layout(self):
        layout = [[sg.pin(sg.Graph(
            canvas_size=(self.width, self.height),
            graph_bottom_left=(0, 0),
            graph_top_right=(self.width, self.height),
            key='-IMAGE-',
            background_color='white',
            change_submits=True,
            enable_events=True,
            drag_submits=True))],
            [sg.Button('Show Image', key='-IMG-'), sg.Button('Show Edges Detected', key='-EDGE-'), 
             sg.Button('Show Edges on Image', key='-CONT-'), sg.Button('Show Mask', key='-MASK-'),
             sg.Button('Reset', key='-RESET-')],
             [sg.Button('Only Threshold', key='-THRESH-', disabled=self.only_threshold),sg.Button('Canny Edge Detection', key='-CANNY-', disabled=(not self.only_threshold))]]
        return layout
    
    def create_window(self):
        image_data = np_im_to_data(self.image)
        window = sg.Window('Display Image', self.layout, finalize=True)    
        window['-IMAGE-'].draw_image(data=image_data, location=self.before_loc)
        return window
    
    def initMask(self):
        gray = cv2.cvtColor(self.originalImage, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (7,7), cv2.BORDER_DEFAULT)
        CannyAccThresh, edges = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        if (not self.only_threshold):
            CannyThresh = 0.1 * CannyAccThresh;
            edges = cv2.Canny(blur,CannyThresh,CannyAccThresh, L2gradient=True)
        contours, hier = cv2.findContours(edges, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        mask = np.zeros(blur.shape, dtype=np.uint8)
        for i in range(len(contours)):
            mask = cv2.fillConvexPoly(mask, contours[i], i)
        mask.tofile('save.csv', sep=',', format='%s')
        return edges, contours, mask
    
    def update_GUI(self, new_image):
        self.image = new_image
        self.layout = self.create_layout()
        self.window.close()
        self.window = self.create_window()

    def remove(self, e):
        tempMask = np.array(cv2.fillConvexPoly(np.zeros(shape=(self.height, self.width)), self.contours[self.mask[e.y, e.x]], 255), dtype=np.uint8)
        self.altered_image = cv2.inpaint(src=self.image, inpaintMask=tempMask, inpaintRadius=3, flags=cv2.INPAINT_NS)
        self.update_GUI(self.altered_image)
    
    def display_image(self):
        self.focus = True
        self.update_GUI(self.altered_image)
    def display_edges(self):
        self.focus = False
        self.update_GUI(self.edges)
    def display_contours(self):
        self.focus = False
        image_copy = self.originalImage.copy()
        cv2.drawContours(image=image_copy, contours=self.contours, contourIdx=-1, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
        self.update_GUI(image_copy)
    def display_mask(self):
        self.focus = False
        self.update_GUI(self.mask)
    def reset(self):
        self.focus = True
        self.update_GUI(self.originalImage)

    def change_methods(self):
        self.edges, self.contours, self.mask = self.initMask()
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
        
        if event == '-IMAGE-+UP' and im.focus==True:
            e = im.window.user_bind_event
            im.remove(e)
        if event == '-IMG-':
            im.display_image()
        if event == '-EDGE-':
            im.display_edges()
        if event == '-CONT-':
            im.display_contours()
        if event == '-MASK-':
            im.display_mask()
        if event == '-RESET-':
            im.reset()
        if event == '-THRESH-':
            im.only_threshold = True
            im.change_methods()
            im.window['-CANNY-'].update(disabled=False)
            im.window['-THRESH-'].update(disabled=True)
            im.reset()
        if event == '-CANNY-':
            im.only_threshold = False
            im.change_methods()
            im.window['-CANNY-'].update(disabled=True)
            im.window['-THRESH-'].update(disabled=False)
            im.reset()
        if event == sg.WINDOW_CLOSED or event == 'Quit':
            break

def main():
    parser = argparse.ArgumentParser(description='A simple image viewer.')

    parser.add_argument('file', action='store', nargs='?', const=0, default=None, help='Image file.')
    args = parser.parse_args()
    display_image(args.file)
    
if __name__ == '__main__':
    main()
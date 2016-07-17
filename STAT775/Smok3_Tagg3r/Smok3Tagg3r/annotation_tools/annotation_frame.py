import tkinter
import tkinter.filedialog
import tkinter.constants
import annotation_tools.annotation_canvas as annotation_canvas


UNR_LOGO_FILE_NAME = 'C:/Users/Terence/Pictures/unr.jpg'
FIRE_FILE_NAME = 'C:/Users/Terence/Desktop/Homewood_BGsmokey.050_10.d/Homewood_BGsmokey.050_10.d/image-2019.jpeg'
OPENING_MESSAGE_IMAGE_FILE_NAME = 'resources/open_image.png'


class AnnotationFrame(tkinter.Frame):
    def __init__(self, parent):
        tkinter.Frame.__init__(self, parent)

        self.config(background = 'silver')

        self.short_directions = tkinter.Label(self,
                      text='Select the areas of interest in the image below. (red == area of interest; yellow == not so much)',
                      bg='white')

        self.image_label = tkinter.Label(self, text='', bg='white')
        self.button_grid = tkinter.Frame(self)
        # in classes that extend this one, don't forget to bind appropriate events to the canvas for your purposes -
        # it is only initialized here
        self.canvas = annotation_canvas.AnnotatedCanvas(master=self, width=annotation_canvas.DEFAULT_IMAGE_CANVAS_WIDTH,
                                                        height=annotation_canvas.DEFAULT_IMAGE_CANVAS_HEIGHT,
                                                        cursor='crosshair', relief=tkinter.RAISED, bd=8)

        self.short_directions.grid(row = 0, column = 0)
        self.canvas.grid(row = 1, column = 0)
        self.button_grid.grid(row = 1, column = 1)
        self.image_label.grid(row = 2, column = 0)

    # def show_unr_logo(self):
    #     self.canvas.set_image(UNR_LOGO_FILE_NAME)



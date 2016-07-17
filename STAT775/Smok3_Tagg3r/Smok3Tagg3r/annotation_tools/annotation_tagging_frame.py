import json
import tkinter
import tkinter.filedialog
import os.path

import annotation_tools.annotation_frame as annotation_frame


TAGGING_MODE_OPENING_MESSAGE_IMAGE_FILE_NAME = 'resources/open_image.png'


class AnnotationTaggingFrame(annotation_frame.AnnotationFrame):
    """ A frame for an AnnotationCanvas for tagging initial areas of interest. """

    def __init__(self, parent):
        annotation_frame.AnnotationFrame.__init__(self, parent)

        self.image_name = TAGGING_MODE_OPENING_MESSAGE_IMAGE_FILE_NAME

        self.short_directions.config(
            text='Tag the interesting areas by clicking, dragging, and releasing to put an annotation on the canvas.')

        self.button_grid = tkinter.Frame(self)
        self.button_grid.grid(row=1, column=1)

        button_opt = {'width': 20, 'padx': 5, 'pady': 5}
        tkinter.Button(self.button_grid, text='Get a New Image to Tag', command=self.get_new_image,
                       **button_opt).grid(row=0, column=0)
        tkinter.Button(self.button_grid, text='Save Annotations To File', command=self.save_annotations,
                       **button_opt).grid(row=1, column=0)
        tkinter.Button(self.button_grid, text='Undo Last Tag', command=self.remove_last_annotation, **button_opt).grid(
            row=2, column=0)

        # just for funsies
        # tkinter.Button(self.button_grid, text='show_unr_logo', command=self.show_unr_logo, **button_opt).grid(row=0,
        # column=1)

        self.canvas.bind_tagging_single_click_start()
        self.canvas.bind_tagging_single_click_release()
        self.canvas.bind_tagging_click_hold()

        self.canvas.set_image(self.image_name)

    def get_new_image(self):
        self.image_name = tkinter.filedialog.askopenfilename()
        self.image_label.config(text=os.path.basename(self.image_name))
        self.canvas.set_image(self.image_name)

    def save_annotations(self):
        save_file_name = tkinter.filedialog.asksaveasfilename()
        if not save_file_name.lower().endswith('.json'):
            save_file_name = save_file_name + '.json'

        annotations = self.canvas.get_selected_annotations_json()

        json.dump({'image_file': os.path.basename(self.image_name), 'annotations': annotations},
                  fp=open(save_file_name, 'w'), indent=4)

    def remove_last_annotation(self):
        self.canvas.delete_last_annotation()

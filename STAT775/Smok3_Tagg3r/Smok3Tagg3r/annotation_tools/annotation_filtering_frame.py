import json
import os
import tkinter
import tkinter.filedialog
from annotation_tools import annotation

import annotation_tools.annotation_frame as annotation_frame


FILTER_MODE_OPENING_MESSAGE_IMAGE_FILE_NAME = 'resources/open_image.png'

class AnnotationFilteringFrame(annotation_frame.AnnotationFrame):
    """ A frame for an AnnotatedCanvas that can be used to select annotations as interesting or not."""

    def __init__(self, parent):
        annotation_frame.AnnotationFrame.__init__(self, parent)

        self.image_name = FILTER_MODE_OPENING_MESSAGE_IMAGE_FILE_NAME

        self.short_directions.config(text = 'Select the areas of interest in the image below. (red == area of interest; yellow == not so much)')

        button_opt = {'width': 30, 'padx': 5, 'pady': 5}
        tkinter.Button(self.button_grid, text='Get another Annotation/Image Pair', command=self.get_new_annotated_image,
                       **button_opt).grid(row=0, column=0)
        tkinter.Button(self.button_grid, text='Save Red Annotations', command=self.save_kept_annotations,
                       **button_opt).grid(row=1, column=0)
        tkinter.Button(self.button_grid, text='Save Yellow Annotations', command=self.save_rejected_annotations,
                       **button_opt).grid(row=2,
                                          column=0)
        # # just for funsies
        # tkinter.Button(self.button_grid, text='show_unr_logo', command=self.show_unr_logo, **button_opt).grid(row=1,
        #                                                                                                    column=0)

        self.canvas.bind_filter_single_click_action()
        self.canvas.set_image(self.image_name)

        self.annotation_file = ''
        self.image_working_dir = ''
        self.dir_opt = options = {}
        options['mustexist'] = False
        options['parent'] = parent
        options['title'] = 'Select the FOLDER where the image the annotation refers to is located (you do not need to find the image file itself).'

    def get_new_annotated_image(self):
        annoation_file_name = tkinter.filedialog.askopenfilename()
        image_dir = tkinter.filedialog.askdirectory(**self.dir_opt)

        annotation_json = json.load(fp=open(annoation_file_name, 'r'))
        self.image_name = annotation_json['image_file']

        self.canvas.set_image(os.path.join(image_dir, self.image_name))
        for json_annotation in annotation_json['annotations']:
            self.canvas.add_annotation(annotation.Annotation().from_json(json_annotation))

        self.image_label.config(text=self.image_name)

    def save_kept_annotations(self):
        save_file_name = tkinter.filedialog.asksaveasfilename()
        if not save_file_name.lower().endswith('.json'):
            save_file_name = save_file_name + '.json'

        annotations = self.canvas.get_selected_annotations_json()

        json.dump({'image_file': self.image_name, 'annotations': annotations}, fp=open(save_file_name, 'w'), indent=4)

    def save_rejected_annotations(self):
        save_file_name = tkinter.filedialog.asksaveasfilename()
        if not save_file_name.lower().endswith('.json'):
            save_file_name = save_file_name + '.json'

        annotations = self.canvas.get_non_selected_annotations_json()

        json.dump({'image_file': self.image_name, 'annotations': annotations}, fp=open(save_file_name, 'w'), indent=4)
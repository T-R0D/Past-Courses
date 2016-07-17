import annotation_tools.annotation as annotation

__author__ = 'Terence'

import tkinter as tk
from PIL import ImageTk, Image

DEFAULT_IMAGE_CANVAS_WIDTH = 1000  # approx. 1920 / 2
DEFAULT_IMAGE_CANVAS_HEIGHT = 575  # approx. 1080 / 2

TAGGING_MODE = 'tag'
FILTER_MODE = 'filter'

class AnnotatedCanvas(tk.Canvas):
    def __init__(self, master=None, mode = FILTER_MODE, cnf={}, **kw):
        tk.Canvas.__init__(self, master=master, cnf=cnf, **kw)
        self.annotations = []
        self.annotation_ids = []
        self.image = None
        self.scale_factor = 1

        self.annotation_start = None
        self.annotation_stop = None
        self.intermediate_rectangle = None

    def set_image(self, file_name):
        img = Image.open(file_name)
        width = img.size[0]
        height = img.size[1]
        while ((width / self.scale_factor) > DEFAULT_IMAGE_CANVAS_WIDTH or (height / self.scale_factor) > DEFAULT_IMAGE_CANVAS_HEIGHT):
            self.scale_factor += 1
        img = img.resize((int(width / self.scale_factor), int(height / self.scale_factor)))
        self.image = ImageTk.PhotoImage(master=self, image=img)
        self.create_image(0, 0, anchor=tk.NW, image=self.image)

    def add_annotation(self, p_annotation):
        if isinstance(p_annotation, annotation.Annotation):
            self.annotations.append(p_annotation)
            self.annotation_ids.append(self.paint_annotation(p_annotation))
        else:
            raise Exception('Let\'s use Annotation class for now')  # TODO: use subclass of Exception

    def delete_last_annotation(self):
        if len(self.annotation_ids) > 0:
            self.delete(self.annotation_ids[-1])
            self.annotation_ids.pop()
            self.annotations.pop()

    def paint_annotation(self, annotation):
        return self.create_rectangle(annotation.start_x() / self.scale_factor, annotation.start_y() / self.scale_factor,
            annotation.stop_x() / self.scale_factor, annotation.stop_y() / self.scale_factor,
            outline=annotation.get_color(), width=2, fill='')

    def repaint_annotations(self):
        for id in self.annotation_ids:
                self.delete(id)
        for annotation in self.annotations:
                self.annotation_ids.append(
        self.paint_annotation(annotation))

    def get_selected_annotations_json(self):
        filtered_annotations = filter(lambda x: x.get_color() == annotation.HIT_COLOR, self.annotations)
        return [annotation.to_dict() for annotation in filtered_annotations]

    def get_non_selected_annotations_json(self):
        filtered_annotations = filter(lambda x: x.get_color() != annotation.HIT_COLOR, self.annotations)
        return [annotation.to_dict() for annotation in filtered_annotations]

    def bind_filter_single_click_action(self):

        def action(event):
            # TODO: use .find_closest() method for cleaner code
            annotation_index = -1
            for i in range(0, len(self.annotations)):
                annotation = self.annotations[i]
                if (annotation.start_x() / self.scale_factor < event.x and annotation.start_y() / self.scale_factor < event.y and
                    event.x < annotation.stop_x() / self.scale_factor and event.y < annotation.stop_y() / self.scale_factor):
                    annotation_index = i
            if annotation_index != -1:
                annotation = self.annotations[annotation_index]
                self.delete(self.annotation_ids[annotation_index])
                self.annotations[annotation_index].rotate_to_next_color()
                self.annotation_ids[annotation_index] = self.paint_annotation(annotation)

        self.bind('<Button>', action)

    def bind_tagging_single_click_start(self):

        def action(event):
            self.annotation_stop = None
            self.annotation_start = (event.x, event.y)

        self.bind('<Button-1>', action)


    def bind_tagging_single_click_release(self):

        def action(event):
            self.annotation_stop = (event.x, event.y)

            self.delete(self.intermediate_rectangle)
            self.intermediate_rectangle = None

            x = min(self.annotation_start[0], self.annotation_stop[0]) * self.scale_factor
            y = min(self.annotation_start[1], self.annotation_stop[1]) * self.scale_factor
            width = abs(self.annotation_start[0] - self.annotation_stop[0]) * self.scale_factor
            height = abs(self.annotation_start[1] - self.annotation_stop[1]) * self.scale_factor

            self.add_annotation(annotation.Annotation().from_args(x = x, y = y, width = width, height = height).make_interesting())

        self.bind('<ButtonRelease-1>', action)

    def bind_tagging_click_hold(self):

        def action(event):
            if self.intermediate_rectangle is not None:
                self.delete(self.intermediate_rectangle)
            self.intermediate_rectangle = self.create_rectangle(self.annotation_start[0], self.annotation_start[1], event.x, event.y, outline='gray', fill = '', dash = (5, 5))

        self.bind('<B1-Motion>', action)
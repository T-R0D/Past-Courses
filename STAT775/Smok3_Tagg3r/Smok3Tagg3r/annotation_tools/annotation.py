import json

__author__ = 'Terence'

HIT_COLOR = 'red'  # aka where there's smoke
MISS_COLOR = 'yellow'  # aka where there's nothing interesting


class Annotation:
    def __init__(self):
        self.upper_left_x = 0
        self.upper_left_y = 0
        self.width = 1
        self.height = 1
        self.color = MISS_COLOR

    def from_args(self, x, y, width, height):
        self.__init__()
        self.upper_left_x = x
        self.upper_left_y = y
        self.width = width
        self.height = height
        return self

    def from_json(self, json_annotation):
        self.upper_left_x = json_annotation['x']
        self.upper_left_y = json_annotation['y']
        self.width = json_annotation['width']
        self.height = json_annotation['height']
        self.color = MISS_COLOR
        return self

    def to_dict(self):
        return {'x': self.upper_left_x, 'y': self.upper_left_y, 'width': self.width, 'height': self.height}

    def start_x(self):
        return self.upper_left_x

    def start_y(self):
        return self.upper_left_y

    def stop_x(self):
        return self.upper_left_x + self.width

    def stop_y(self):
        return self.upper_left_y + self.height


    def get_color(self):
        return self.color

    def rotate_to_next_color(self):
        if self.color == MISS_COLOR:
            self.color = HIT_COLOR
        elif self.color == HIT_COLOR:
            self.color = MISS_COLOR
        else:
            self.color = MISS_COLOR

        return self

    def make_interesting(self):
        self.color = HIT_COLOR
        return self
import base64
import logging
import tempfile
import tkinter
import tkinter.messagebox
import zlib

import annotation_tools.annotation_filtering_frame as filtering_frame
import annotation_tools.annotation_tagging_frame as tagging_frame

FILTER_MODE = 'filter'
TAG_MODE = 'tag'

class Application(tkinter.Frame):
    def __init__(self, master=None):
        self.logger = logging.getLogger('main_module')

        tkinter.Frame.__init__(self, master)

        self.master.title('Smok3 Tagg3r')
        self.master.iconbitmap(default=TRANSPARENT_ICON_PATH)

        self.master.resizable(width=False, height=False)
        self.master.geometry('+50+50')

        self.config(background = 'silver')

        self.grid()

        self.menu_parent = tkinter.Menu(master=self)

        self.mode = TAG_MODE

        file_menu = tkinter.Menu(self.menu_parent, tearoff=0)
        file_menu.add_command(label='Exit', command=self.quit)

        mode_menu = tkinter.Menu(self.menu_parent, tearoff=0)
        mode_menu.add_command(label="Tagging Mode", command=self.switch_mode)
        mode_menu.add_separator()
        mode_menu.add_command(label="Filtering Mode", command=self.switch_mode)

        help_menu = tkinter.Menu(self.menu_parent, tearoff=0)
        help_menu.add_command(label='Using TAGGING mode', command=self.show_tag_mode_info)
        help_menu.add_command(label='Using FILTERING mode', command=self.show_filter_mode_info)
        help_menu.add_separator()
        help_menu.add_command(label='About...', command=self.show_about_info)
        help_menu.add_command(label='License', command=self.show_license)

        self.master.config(menu=self.menu_parent)
        self.menu_parent.add_cascade(label='File', menu=file_menu)
        self.menu_parent.add_cascade(label='Mode', menu=mode_menu)
        self.menu_parent.add_cascade(label='Help', menu=help_menu)

        self.init_tagging_mode()

    def switch_mode(self):
        widgets = self.winfo_children()
        for widget in widgets:
            if widget != self.menu_parent:
                widget.destroy()

        if self.mode == FILTER_MODE:
            self.mode = TAG_MODE
            self.init_tagging_mode()
        elif self.mode == TAG_MODE:
            self.mode = FILTER_MODE
            self.init_filter_mode()
        else:
            self.mode = FILTER_MODE
            self.init_filter_mode()

    def init_filter_mode(self):
        self.logger.info('Switching to filter mode')
        tkinter.Label(self, text='FILTER MODE', bg='light blue', fg='black', font='Times 20 bold').grid(row=0, column=0)
        filtering_frame.AnnotationFilteringFrame(parent = self).grid(row = 1, column = 0)

    def init_tagging_mode(self):
        self.logger.info('Switching to tagging mode')
        tkinter.Label(self, text='TAGGING MODE', bg='light green', fg='black', font='Times 20 bold').grid(row=0,
                                                                                                          column=0)
        tagging_frame.AnnotationTaggingFrame(parent = self).grid(row = 1, column = 0)

    def show_about_info(self):
        tkinter.messagebox.showinfo(title='About', message=ABOUT_MESSAGE)

    def show_license(self):
        tkinter.messagebox.showinfo(title='License', message=LICENSE_MESSAGE)

    def show_usage_info(self):
        tkinter.messagebox.showinfo(title='Usage', message=USAGE_MESSGAE)

    def show_tag_mode_info(self):
        tkinter.messagebox.showinfo(title='Tagging Mode', message=TAG_MODE_HELP_MESSAGE)

    def show_filter_mode_info(self):
        tkinter.messagebox.showinfo(title='Filtering Mode', message=FILTER_MODE_HELP_MESSAGE)

# program resources
TRANSPARENT_ICON = zlib.decompress(base64.b64decode('eJxjYGAEQgEBBiDJwZDBy'
                                                    'sAgxsDAoAHEQCEGBQaIOAg4sDIgACMUj4JRMApGwQgF/ykEAFXxQRc='))

_, TRANSPARENT_ICON_PATH = tempfile.mkstemp()
with open(TRANSPARENT_ICON_PATH, 'wb') as icon_file:
    icon_file.write(TRANSPARENT_ICON)

ABOUT_MESSAGE = \
"""
This program was written in Python 3.4.3 using the tkinter GUI library.

This program uses the GPL v2.0 software license (for now).

This program was created by Terence Henriod, who can offer
minimal (and probably delayed) support if you contact him at
t-r0d@hotmail.com (that's a zero in 't-r0d').
(But at least there's an offer for support)
"""

LICENSE_MESSAGE = \
"""
Copyright (c) 2015,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of Smok3 Tagg3r nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

USAGE_MESSGAE = \
"""
How do I use this? Instructions are under development.
"""

TAG_MODE_HELP_MESSAGE = \
"""
Tagging mode is for creating fresh, new labels or annotations for an
image manually. Use this mode when you want a human to directly
identify where areas of interest are in an image, such as where a face
or smoke is.

[1] First you will need to get an image to tag. Click the
    `Get a New Image to Tag', select your image, and it is as easy as
    that. Do note that your image may appear as a scaled down version
    to fit in the window, but don't worry, the annotation data that is
    stored will be scaled back up when it is stored to match the
    original image.

[2] To create an annotation, simply click and drag to create a
    rectangle around the area of interest, and then release once the
    ectangle is properly drawn. While you are dragging and resizing
    your rectangle, a dashed rectangle should appear to guide you
    as you tag.

[3] But what if you messed up and drew a sloppy annotation? Don't
    worry, the `Undo Last Tag' button is there for you. Each time you
    click that button, the most recently added annotation will be
    removed and deleted. You can actually remove all annotations this
    way.

[4] Once you have drawn the sufficient annotations, you should save
    them to a file. On the right side there is a button labeled
    `Save Annotations To File'. Use it. You will be presented with a
    prompt asking you where to save the file. You can name the file
    anything, and preferably with the extension .json, but if you don't
    put .json, Smok3 Tagg3r will do so for you.
"""

FILTER_MODE_HELP_MESSAGE = \
"""
Filter mode is used to identify good and bad annotations. This mode
was actually the first mode of the project developed because it might
be the most practical. Filter mode was created with the intent that
labeled data could be `bootstrapped'; that is, using preliminary methods,
we can have a computer find areas of interest, and then reduce the
human workload by having just a few areas for the human to consider
rather than the whole image. It also allows us to identify images that
are tricky for a classifier, so we can actually focus our trainging
of classifiers to perform better on tricky inputs.

[1] Start by clicking `Get Another Annotation/Image Pair'. This will
    present you with two dialog boxes, one after the other. The first
    dialog box wants you to locate an existing annotation file
    (likely a .json file). The second wants you to locate just
    the directory that the image file that corresponds to the
    annotation file is in. Smok3 Tagg3r can find the appropriate
    image file on its own at that point. Your use case might be
    different, but in the case that inspired the creation of Smok3
    Tagg3r, this eliminates the need for a user to manually sift
    through thousands of image files that are contained in a folder.
    Again, the image appears, possibly scaled down. The annotations
    from the given file also appear n the image in yellow. Supplying
    files of the incorrect type will produce an error that the
    application will likely just ignore and you will have to try again.

[2] Now it's time to filter! Simply click the annotations that are
    `good' and they will turn red. If you change your mind, just
    click that annotation again and it will turn back to yellow.

[3] When finished, be sure to save your annotations. Use the
    appropriate button, `Save <color> Annotations', to save your
    annotations. Try to use appropriate names; it is recommended to
    save both sets. Again, you can add the .json extension to the
    names, or you can let it be added for you.
"""
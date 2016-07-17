#!/usr/bin/python3

from main_module.application import Application

TEST_NUM_REGIONS = 4
UNR_LOGO_FILE_NAME = 'C:/Users/Terence/Pictures/unr.jpg'
FIRE_FILE_NAME = 'C:/Users/Terence/Desktop/Homewood_BGsmokey.050_10.d/Homewood_BGsmokey.050_10.d/image-2019.jpeg'
TEST_JSON_FILE_NAME = 'C:/Users/Terence/Documents/GitHub/STAT775/GuiProject/z_test_resources/test_annotations00.json'

def main():
    my_app = Application()
    my_app.mainloop()

if __name__ == '__main__':
    main()
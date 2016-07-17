# Smok3 Tagg3r

**For quickly annotating areas of interest in images.**

## What is it?
Smok3 Tagg3r is a lightweight gui application for tagging images for computer vision oriented applications. Users can use it to tag new areas of interest by hand, or use Smok3 Tagg3r to filter out locations identified by preliminary methods. Instead of slicing out 'sub-images' or anything like that, Smok3 Tagg3r uses simple rectangular annotations to indicate where areas of interest are. The annotations are stored using simple, human-readable JSON.

## What's with the name?
Smok3 Tagg3r is written in Python**3**. The inspiration/motive for the project was a final class project where everyone worked together to use computer vision methods to detect forest fires around scenic Lake Tahoe by looking for the earliest smoke plumes. My part in this effort was to create an application that would reduce the human effort required to label our data.

## Will this project be actively maintained?
Doubtful. That being said, I am very open to pull requests, and may be willing to help you out if you want/need a quick/easy feature. This is not my main focus, it is intended to be quick, easy, and helpful, but nothing of enterprise quality.

## What's this written in?
 Smok3 Tagg3r is written exclusively in Python3 and currently uses tkinter for the GUI stuff because it ships with most Python distributions, is cross-platform, and has minimal dependencies.
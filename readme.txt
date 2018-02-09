# go to build directory
$ mkdir build & cd build

# generate Makefile using cmake
$ cmake ..

# compile the code
$ make

# checkout the usage
$ ./opencv_face_align
# Incorrect command line input! 
# Usage: opencv_face_align facial_patch 5pt_anno rot_patch rescale_patch crop_patch

# up and run ... 
$ ./opencv_face_align 9_040.jpg 9_040.jpg.txt 9_040_rot.jpg 9_040_rescale.jpg 9_040_crop.jpg
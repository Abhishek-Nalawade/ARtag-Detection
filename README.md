# ARtag-Detection

## Use the link below to download the input data
https://drive.google.com/drive/folders/1gqKuZ38cdTCLLTmuh_AFrqzs1JMdUUUB?usp=sharing

## Details
This is the input frame

<img src ="Output/Multiple%20Tags%20Input.PNG" width ="500">

The ARtag's are warped and bilinearly interpolated to fill out the holes in the warped ARtag. Further the image is applied a threshold to
improve the quality for decoding the tag ID

<p float="left">
<img src ="Output/Warped%20ARtag.PNG" width ="100">
<img src ="Output/Warped%20and%20interpolated%20ARtag.PNG" width ="100">
<img src ="Output/Threshold%20ARtag.PNG" width ="100">
</p>

Finally an image is overlaped on the ARtag according to the orientation of the tag

<img src ="Output/Multiple%20Tags%20Output.PNG" width ="500">


## Libraries Required
* Numpy
* OpenCV
* Scipy
* Matplotlib
* imutils

## Instructions:
1) To run all the programs make sure to download the required data first and have them in the same directory as the code.
2) The video can be saved by uncommenting the proper lines from the code.

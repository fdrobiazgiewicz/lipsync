Engineering Thesis skeleton [Work in progress]
* rewrite it in TeX instead of markdown
* start preparing a bibtex

# Questions
* Is it possible to make this project ffmpeg agnostic?
* How to extract YouTube videos from a channel?

# Problem statement
The purpose of this excercise is to prepare video sequences or components for a machine learning algorithm to detect the audio-video shift by (hopefully) "reading the lips". As machine learning, especially deep learning requires lots of unbiased data for both training and verification. The program/framework developed here in this excercise should provide the following functionalities.
* generate fixed shift between audio and video up to N milliseconds in both ways
* add some artifiacts
* freeze some frames
* drop some frames
* [maybe] change the playback speed for audio/video - for example linearly with configurable parameter

# Datasources
In this thesis, the primary (and supposedly only) datasource used will be *YouTube*.
It's almost sure that the framerate for *downloaded videos* will be fixed and equal to 30 FPS. Videos with 50 FPS should be processable as well thanks to downscaling.
It would be nice to be able to pass the name of the whole YouTube channel and an optional number of wanted (desired) videos.
* CNN channel (preferably lots of talking heads)
* lifestyle youtube channel - one person talking to the camera, close to no music, close to no cuts
* pranks youtube channel - not much content whatsoever possibly many people, many cuts
* movie trailers - not much talking, many people, many cuts
* twich - streaming applications, gaming
Obtaining vide IDs for the whole channel can be possibly done using the YouTube API or web scraping techniques i.e. extracting the links from the raw html file. The latter can seem not so elegant, however sometimes it's the only solution.
* Worth checking out - regex for YouTube video id.
* https://github.com/HermanFassett/youtube-scrape this one can be interesting, but needs to be deployed on Heroku.
* Discard videos marked as non-english as the target project should learn from English only content? [discuss]
* Check if it's possible to download grayscale in order to minimize network traffic.

# Modifying
* normalize and align the faces to then apply filters (artifacts around the mouth)

## Frames
* hold the frame for N seconds/frames
* produce encoding artifacts around the mouth
* skip some frames - Dropped framerate can occur along with streaming.

# Brainstorming

## Naive approach
The first thing that comes to my mind [change that - try to write in an impersonal form] is preparing the dataset using a video editing software such as *Sony Vegas*, *Adobe Premiere*, *FinalCut*. Of course initial video sequences are to be downloaded from *YouTube*. This way one can have full control over audio/video - check it and adjust "on the fly" with no need for rendering. What is important as well, is the fact that this way **a human can tell if introduced latency is visible/audible and affects the perceived quality (in terms of A/V latency).**
In this case the human factor is both the advantage and disadvantage.
On one side it provides the much needed human insight and instant feedback if the modifications are sufficient. On the other side human work makes the process slow - of course there are some limited scripting possibilities in those programs.
Although it's possible to acomplish - this software is supposed to facilitate other more advanced processes such as color grading, post-processing and other editing techniques.

## Semi automated
Searching for videos by hand and then running the automatic script.
+ simple
+ doesn't involve YouTube scraping/API usage
+ reduced amount of human editing work
- searching for videos by hand (you have to check by yourself if there is more than one face and so on)

## Ideal solution [TODO: mark some features as good to have]
* automated downloading
* process grayscale content as information is located mostly in luminance
* stream processing - easy on the disk, process data as it comes
* normalize video - downscale the resulting resolution
* when face is not detected, "keep the momentum" and run for N frames more
* the above would be useful with hand movements around the face (such as fixing hair)
* configure output format (resolution, framerate - maybe not needed, file extensions, encoding)
* Discard video with more than one face detected. This may too early for the machine learning to deal with more than one face in the video.
* lazy - reduce used disk space by creating modified video files on the go. Consider passing OpenCV/numpy objects instead of video files. This should be good for machine learning as it skips reading the file back into opencv/numpy objects.

# Software tools
The framework for this thesis has following dependencies:
* pytube - python module for YouTube video downloading
* ffmpeg - ? (check the version on Mac, try to run it on a designated Ubuntu machine)
* pymovie? - no, this is for lightcurves extraction
* OpenCV - python library built in C++ designed for computer vision operations
* Haar cascades for face detection
* facial landmarks files for normalization and alignment
* numpy - python module for numeric purposes (especially matrix related tasks)
* sk-video - scikit-video, video editing, check, this might be interesting, although this may be the same as opencv.
* moviepy - video editing
* bufferer - fake buffering events https://pypi.org/project/bufferer/ Can this be used to desynchronize audio from video?
* PIL/pillow - this seems to be for still images only, but something may be useful
* pydub - audio-related processing, https://github.com/jiaaro/pydub, operates with milliseconds resolution. This should be enough to keep the content in sync.  With 30 FPS, each frame is 33.3 ms, with 24 FPS - 41.4 ms.
Many of this seem to be ffmpeg wrappers.

MoviePy
* not intended for frame-by-frame work such as face detection
* audio extraction
* audio/video manipulation shifting and playback speed
* adding overlays
* load only needed function in "production" version to reduce application footprint

# Embrace the pipeline processing model
* analogous to map-filter-reduce
* easy to parallelize
* good vertical scalability

# Face detection vs face tracking
* Detection may be costly and running it for every frame seems redundant.
* It may be better to run face detection every few frames and track the face in frames between.

# Notes from writing actual software
For all exercises in this section one video was found on YouTube and downloaded to read it from the disk instead of downloading it everytime. Apart from that, the video was trimmed to roughly 5 seconds to reduce the processing time. This enabled me to investigate different approaches to face detection.

## haar cascade only
This one is a simple example straight from the OpenCV hands-on tutorial.
1. Read the file frame by frame, which is a simple *while loop* syntax.
2. Convert the frame to grayscale.
3. Apply processing to each frame i.e. Haar cascade based face detection.
4. Method above returns coordinates and dimensions of a box with found face (double check that). Those are used to cut the face from the frame.
5. Cut face is appended to a resulting file.
+ provided xml file with haar cascade
+ self-explanatory method names
- no logic for when the face is not detected
- resulting video is not normalized
- resulting video is "jumpy" - facial landmarks are located in different places in different frames
Questions:
* Is it needed to run the detection method in each frame - is it possible to run it every N frames and just track it in between?
* How well does it perform with other, external xml files?

## hog face alignment
* Histogram of Oriented Gradients - what does it mean?
* dlib - C++ library
* showing each frame takes quite a long time - hard for debugging
* Writing to a file takes a long time. This may be an issue with ffmpeg.
* The chin is visible - there is some marigin around the face.

## usability_checker.py
1. Download a YouTube video.
2. Apply count faces.
3. Calculate usability based on the following
    * number of parts with consecutive 1s - chunks where one face is visible
    * number of parts with consecutive 0s (or not 1s) - intros and so on
- Has to download a video to process it, thus is highly available on the network.
* Ideally use streaming and process it frame-by-frame, drop the connection when the video is not usable (some 
threshold).
* This should produce a metric and suggest timestamps/frames which are "worth processing".
* `chunk_length` determines the resolution of sequence boundaries.
* 

## count_faces.py
Process a given video file and produce a vector of number of faces detected in each frame.
The processing logic here doesn't have to be precise. It's meant to roughly assess the usability of given video.
* If there is more than one face (or no face) for a couple of frames (200 ms * 24 FPS that would be 5 frames) 
probably this is an artifact and more advanced processing won't make the same mistake.
* It seems that using mean and standard deviation is enough.
* Using `len(detected_faces)` induced problems with mean as `mean([0,0,2,2]` is the same as `mean([1,1,1,1]`, yet the
 latter is significantly better for our purpose.
* Maybe this could be done in parallel? What about frame ordering? This may be a problem with parallel processing.

## scene detection
Another valid point is detecting scenes.
* It may be wise to first detect scenes and process only those worth it (namely those longer than N frames).
* It's worth noting that once we have the video split into scenes (sequeneces of similiar frames) it may be redundant to detect faces in all of frames!
* Threshold value can be deceiving in case of rapid cuts - vide Dune 042 (many faces)
* Treats screen scrolling as very short scenes <- that's quite good as we don't want it

* `save-images --num-images n` produces first, last frame from the scene and n other in between
* analyze pictures to determine which scenes are to be cut out
* Use the `split_video_ffmpeg` method from `video_splitter.py` on list from the point above


# stages
* describe the process in stages `description: input -> output`
    1. download from YT, rename with video id: link -> path to file
    2. analyze shots: path to file -> timestamps of scenes and N frames from each
    3. check N frames if the scene has a face: timestamps of scenes and N frames from each -> timestamps of scenes 
    with faces
    4. extract scenes with faces from the video: timestamps -> 
    
# check this videos
* https://www.youtube.com/watch?v=UnoE8M5qbrk - monologue

# state of art
* face detection filters in instagram/tiktok. Especially tracking & normalization in tiktok.

# cloud based solution
* video processing @clarifai

# state of the art
 * 300 Videos in the Wild (300-VW) Challenge & Workshop (ICCV 2015) - those videos are 1 minute long
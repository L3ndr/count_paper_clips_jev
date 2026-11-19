# Count paper clips in images with Jev (TypeSafe)
## Introduction
### Idea
At university, in the subject "Informatik 2", I had to train a convolutional neural network (CNN) to count paper clips in an image. In that project I first assembled a dataset with a script to have clean labeled data. On this data I designed and trained a CNN to count the number of paper clips in the image. I also counted how many paper clips of each color there are. (https://github.com/L3ndr/KIIB-Bueroklammer-CNN)

### Goal
This project aims to recreate this assignment using Jev, a new model by [TypeSafe](https://typesafe.ai/), instead of a CNN to determine the number of paper clips in the image. I will be writing a script for creating images as input, similar to the dataset creation in the original project. It is followed by a pipeline for classifying the images with Jev. To round off the project, I will evaluate the accuracy of this project compared to the original.

### What is Jev?
Jev is TypeSafe's flagship model and the first "System One" model. Unlike LLMs, it doesn't generate text that has to be parsed. Instead, it answers typed questions about a given state (e.g. an image) and returns structured results: typed values, probability distributions and, for some question types, a confidence score.

The question types relevant for this project:

| Question type | Goal | Returns |
|---|---|---|
| Choice | Choose an option from a list (e.g. the number of paper clips) | choice, probabilities, confidence |
| Box | Locate an object in the image | box (top, bottom, left, right), confidence |

Several questions can be sent in one request. They are evaluated in parallel and independently of each other, so it's possible to ask for the total count and the count per color at once.

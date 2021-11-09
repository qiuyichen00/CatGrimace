# CatGrimace

## Introduction:

Cat Grimace is a project that trys to auto-detect if a specific cat in video is in pain. The current approach is to evalute the pain score corresponding to individual action units 
of cats, and add all scores together to get a final pain score, which tells the level of pain the cat is experiencing. The action units includes face, eyes, ears,mouth, cheek, 
and whizzle.

## Current Progress:

Implemented YOLO-V3 object detection model to predict the bounding boxes for each action unit.

## Sample Prediction
![] (/sample_pred.png)


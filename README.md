#ADL_HW1
#Task1 Intent classification
## Description
This is pytoch implementation for the task of sentence classification using RNN.<br>
There is 150 of classes represent differenct kind of main topic according to each sentence.<br>
I have 15000 of sentence prepare for training data. <br>
Using Glove dictionary embedding vector into all words, incase it took long time and memory to process 
Glvoe dictionary we cloud also creat our own dictionary depands on what we need. 
## Setup
create a new enviroment with Python 3.8:<br>
pip install the text file that class TA give us.<br>
Click "downlod.sh" to downlod all the data we need.
## How to run
training procedure below:
```
open anaconda 
activate the new enviroment I made
cd in the file location
ex: cd C:\Users\LAB228\Desktop\ADL\file_name
type:bash train_cls.sh path${1} path${2}
```
reproduce the model when the train is finish:
```
type:bash train_cls_reproduce.sh path${1} path${2}
```
Remark:
```
${1}:test data ${2}:the csv file I save the test prediction
```
#Task2 slot tags
## Description
This is pytoch implementation for the task of slot tags using RNN.<br>
## How to run
training procedure below:
```
open anaconda 
activate the new enviroment I made
cd in the file location
ex: cd C:\Users\LAB228\Desktop\ADL\file_name
type:bash slot_tags.sh path${1} path${2}
```
reproduce the model when the train is finish:
```
type:bash slot_tags_reproduce.sh path${1} path${2}
```
Remark:
```
${1}:test data ${2}:the csv file I save the test prediction
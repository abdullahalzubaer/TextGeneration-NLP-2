# Generating text using LSTM (word-level)

In this repository I will present how to generate text using LSTM in word level. 

# Dataset and preprocessing:

A novel (Crime and Punishment) in pdf format. The dataset that I have used is a novel and it is in pdf format, extracted and preprocessed taking only ASCII characters and removing numbers/

# Data properties

# Network architecture

```python

model = Sequential([
    Embedding(total_words, 256, input_length=max_sequence_len-1),
    Bidirectional(LSTM(256, return_sequences=True)),
    Dropout(0.01),
    LSTM(256),
    Dense(total_words, activation='softmax')
])


```

# Testing 

(Network trained for 10 epochs)

```
Input Sequence : <START>nly capable of killing. And it seems I wasn't even capable of that . . . Principle? Why was that 
fo<END>

Output Sequence:

people

at in you singing pas his the whisper dress in uproar into in fire silly the market candlestick thread stern

he unable at at children of you it stepping broken proud as back was that ' her over good to

timidly unable and here to the express for weak houses slowly no as an cold the the into at i

the of touched out they thought and the shh to over was a the anyone vahrushin with over the her

it one tears that her and finn dounia sonia's face again on regeneration and you and on in stepping out

and by the you looking and that coloured his and his and disgrace in her a mare in glistened on

over to on gulf question moving her he with thirty so it talking ventured stepping by hurriedly in to without

and stepping garden to of and stout and pictures with there his that afterwards with pas candlestick to a in

educated with and at a my the looked market trunk for the prepared me to her again that till<END>
```
---
## TODO
* [ ] Improve documentation
* [ ] Increase training number of epochs
* [ ] Difference between word and character level language generation
---
Reference: https://www.coursera.org/learn/natural-language-processing-tensorflow



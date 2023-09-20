# GPT2-Lyrics
This is a simple project on how to finetune an LLM model such as OpenAI's GPT2 to be able to recreate songs of a specific artist. To do this, we use as training data all the lyrics of all the songs of the artist, we tokenize them in a specific way so that the model learns to understand and relate them, and then we re-train the model.

In this repository you will find the files that allow you to perform this experiment locally:
- The GPT2_lyrics file (both in its .ipynb and .py versions) contains all the necessary code to read the data file (in this case, it is made for Taylor Swift lyrics), as well as for its preprocessing, model training and inference to see the results obtained.
- The Taylor_swift_lyrics and avicii_lyrics files are two files that contain all lyrics of all songs by both Taylor Swift and Avicii. The code is intended for Taylor Swift lyrics but it is easily extrapolated to any kind of lyrics.

As this project is a mere experiment to create a model at home, I have not uploaded the trained model to any repository for the user to create his own.

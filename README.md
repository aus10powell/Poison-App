# Streamlit app for trying out poison oak classifier
Try it out: https://poison-app.herokuapp.com (note: may take ~30sec for webpage to load since the server is on a non-paid licens)

## What is it
This is a image classifier trained on hand-collected poison oak images over 1 year. Poison ivy will not necessarily expected to be classified as poison oak even though it's leaves also contain the oil causing the allergic reaction, Sumac.

At time of repository creation, this model was implemented in a free iOS app "Poizon Plants", however due to Apple developer licence costs it may not be live.

In order to help the user assess interpret the probability of a given image containing poison oak, different prompts based on the softmax probability are output.

Model is initialized at same time as server start-up to speed up inference. (Tensorflow lite was tested and optimized for latency, but did not significantly improve speed but did significantly decrease f1 metrics)

## Run
*streamlit run app.py*

## Tests
pytest

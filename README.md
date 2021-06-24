# Spruce Budworm Carbon Emissions

### Reporducing Results
We compare results from several models by running `make_dataset.py`. This loads serialized models from a `model` directory and uses them to make predictions on data not seen in training.



### Docker Instructions
To run use the following:
``` 
docker pull sms1097/budworm
docker container run -p 8888:8888 budworm
```

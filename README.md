# ML-Model_Deploy

develop

[![app tests](https://github.com/sevenineone/ML-Model_Deploy/actions/workflows/test.yml/badge.svg?branch=develop)](https://github.com/sevenineone/ML-Model_Deploy/actions/workflows/test.yml)

Simple ml deployment. Model learning using iris dataset.

## Example

You can run it, by `server.py` file or docker. To test it, you can use this comand.

```
curl --location --request POST 'http://0.0.0.0:5000/predict' --header 'Content-Type: application/json' --data-raw '{"sepal_length": 5.1,"sepal_width": 3.5,"petal_length": 1.4,"petal_width":0.2}'
```

## Docker

1) Clone repository.

```
git clone https://github.com/sevenineone/ML-Model_Deploy
```

2) Go to project dir.

```
cd ML-Model_Deploy
```

3) Run docker build

```
docker build -t model_deploy .
```

4) Run container

```
docker run -p 5000:5000 model_deploy
```


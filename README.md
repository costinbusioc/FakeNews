# FakeNews

### System requirements

Python 3.7+ (see installatation steps below)

On Ubuntu 18.04 this means:
```shell
apt-get update -qq
apt-get install -qq build-essential virtualenv python3.7 python3.7-dev
```

### Python dependencies

The requirements.txt and python .venv are shared between the web app and the ontology population.

```shell
virtualenv -p python3.7 .venv
source .venv/bin/activate
pip3 uninstall setuptools && pip3 install setuptools && pip3 install --upgrade pip
pip3 install -r requirements.txt
```

You may also need to install spacy models, using the command:
```shell
python3 -m spacy download model_name
```

For neural coref errors
```shell
pip3 uninstall neuralcoref
pip3 install neuralcoref --no-binary neuralcoref
```

#!/bin/bash

# download streetstyle-27k dataset
wget https://s3.amazonaws.com/kmatzen/streetstyle27k.manifest
wget https://s3.amazonaws.com/kmatzen/streetstyle27k.tar
tar -xvf streetstyle27k.tar

# download NewsAnchor_train dataset
wget https://storage.googleapis.com/esper/tmp/newsAnchor_train_img.tar
wget https://storage.googleapis.com/esper/tmp/newsAnchor_train_manifest.pkl
tar -xvf newsAnchor_train_img.tar


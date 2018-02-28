#!/bin/bash

# download streetstyle-27k dataset
wget https://s3.amazonaws.com/kmatzen/streetstyle27k.manifest
wget https://s3.amazonaws.com/kmatzen/streetstyle27k.tar
tar -xvf streetstyle27k.tar

# download NewsAnchor dataset
gsutil cp gs://esper/tmp/cloth_dict.pkl .
gsutil cp gs://esper/tmp/cloth_test.tar .
tar -xvf cloth_test.tar


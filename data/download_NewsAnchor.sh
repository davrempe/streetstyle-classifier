#!/bin/bash
gsutil cp gs://esper/tmp/cloth_dict.pkl .
gsutil cp gs://esper/tmp/cloth_test.tar .
tar -xvf cloth_test.tar
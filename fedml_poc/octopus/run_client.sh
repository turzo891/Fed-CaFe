#!/usr/bin/env bash

RUN_ID=$1
RANK=$2  # Client rank (e.g., 1 or 2)

python3 client/torch_client.py \
    --cf config/fedml_config.yaml \
    --role client \
    --rank $RANK \
    --run_id $RUN_ID

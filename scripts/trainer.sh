#! /bin/bash

export LOG_LEVEL=info
export PUSH_POLICY_DIR=./push-policy3

MODEL=A3C
WORKERS=35
REWARD_FUNC=3
MANIFEST_DIR="${PUSH_POLICY_DIR}/aft_training"
TRAIN_LOG_FILE="${PUSH_POLICY_DIR}/train.log"
TRAIN_MANIFESTS="$1"

cat ${TRAIN_MANIFESTS} | xargs -I "{}" ./blaze_exec train "{}" \
					--model ${MODEL} \
					--workers ${WORKERS} \
					--use_aft \
					--reward_func ${REWARD_FUNC} \
					--manifest_file "${MANIFEST_DIR}/{}.manifest" \
					--no-resume >> ${TRAIN_LOG_FILE} 2>&1

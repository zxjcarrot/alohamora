[![Build Status](https://travis-ci.com/nkansal96/blaze.svg?branch=master)](https://travis-ci.com/nkansal96/blaze)

# Requirements

1. Install Node.js v10 or higher
2. Install Python 3.6.5 or higher
3. Install virtualenv
4. Install Docker

# Setup

1. Clone the repo
2. `make setup`
3. Download Chrome https://ungoogled-software.github.io/ungoogled-chromium-binaries/releases/linux_portable/64bit/75.0.3770.80-1.2

If you are not already in the virtualenv, run `source .blaze_env/bin/activate`.

# Record Traces
```
cd alohamora
mkdir train_dir
rm -rf train_dir/*
CHROME_BIN=../ungoogled-chromium_75.0.3770.80-1.1_linux/chrome sh scripts/record.sh top10.txt train_dir
```

# Preprocess
```
python scripts/preprocess.py --train_dir train_dir --num_workers=10
```

# Training
```
mkdir -p mkdir push-policy/aft_training
cp train_dir/*.manifest push-policy/aft_training
ls -1 train_dir/*.manifest | sed 's/.manifest//g' > manifests.txt
sh scripts/trainer.sh manifests.txt
```
# Run Tests

`make test`

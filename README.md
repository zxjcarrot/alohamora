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
CHROME_BIN=/home/zxjcarrot/ungoogled-chromium_75.0.3770.80-1.1_linux/chrome sh scripts/record.sh top10.txt train_dir
```

# Preprocess
```
./blaze_exec preprocess --record_dir neversslcom --output manifest_output http://www.neverssl.com
```
# Run Tests

`make test`

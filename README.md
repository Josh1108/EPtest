


### Installation
We require `jiant`. For installation, refer to:
https://github.com/nyu-mll/jiant#installation

Next, run the following commands:

```bash
git clone https://github.com/Josh1108/EPtest/tree/main
pip install -r requirements.txt

# Add the following to your .bash_rc or .bash_profile 
export PYTHONPATH=/path/to/eptest/root/folder:$PYTHONPATH

```

### Running experiments

    .
    ├── converters/              # files to convert data in EPtest fromat
    ├── core/                    # all core functionailites
    │   ├── trainer.py           # training file
    │   ├── test.py              # test file
    │   ├── models/              # models used for EP tests
    |   └── config/              # configs used for training and testing
    └── ...

To train models, run:

```bash
python3 trainer.py --config_file <PATH_OF_FILE>
```

For testing, run:
```bash
python3 test.py --config_file <PATH_OF_FILE>
```

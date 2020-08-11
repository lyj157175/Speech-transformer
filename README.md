# Speech-transformer

## Introduction

This is a PyTorch re-implementation of Speech-Transformer: A No-Recurrence Sequence-to-Sequence Model for Speech Recognition.



## Dataset

Aishell-1



## Performance

Evaluate with 7176 audios in Aishell test set:
```bash
$ python test.py
```

## Results

|Model|CER|Download|
|---|---|---|
|Speech Transformer|11.5|[Link](https://github.com/foamliu/Speech-Transformer/releases/download/v1.0/BEST_checkpoint.tar)|



## Dependency

- Python 3.6.8

- PyTorch 1.3.0

    

## Usage
### Data Pre-processing
Extract data_aishell.tgz:
```bash
$ python extract.py
```

Extract wav files into train/dev/test folders:
```bash
$ cd data/data_aishell/wav
$ find . -name '*.tar.gz' -execdir tar -xzvf '{}' \;
```

Scan transcript data, generate features:
```bash
$ python pre_process.py
```

Now the folder structure under data folder is sth. like:
<pre>
data/
    data_aishell.tgz
    data_aishell/
        transcript/
            aishell_transcript_v0.8.txt
        wav/
            train/
            dev/
            test/
    aishell.pickle
</pre>

### Train
```bash
$ python train.py
```

If you want to visualize during training, run in your terminal:
```bash
$ tensorboard --logdir runs
```

### Demo
Please download the [pretrained model](https://github.com/foamliu/Speech-Transformer/releases/download/v1.0/speech-transformer-cn.pt) then run:
```bash
$ python demo.py
```


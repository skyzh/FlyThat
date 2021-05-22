# FlyBitch
A fork of FlyIt

## Training

```bash
./train.sh --batch 256
```

## Retriving a model to local disk

```bash
./retrieve.sh root@some-server 05-20-trivial-model-02
```

## Evaluate a model

```bash
./evaluate.sh --model saved_runs/05-20-trivial-model-02/runs/model --batch 16
```

## Submit to Kaggle

You need to `pip3 install kaggle` and login first.

```bash
./submit.sh saved_runs/05-20-trivial-model-02/runs  "First submission"
```

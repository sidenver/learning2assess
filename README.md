# A Prioritization Model for Suicide Risk Assessment

This readme describes how to train 3HAN for Task A of CLPsych.

## Step 0. Create Virtual Environment with Miniconda3

### Download Miniconda3

See https://docs.conda.io/en/latest/miniconda.html for download and installation of miniconda3.

For Linux user, after downloading Miniconda3-latest-Linux-x86_64.sh, install miniconda3 by running

```sh
bash Miniconda3-latest-Linux-x86_64.sh
```

### Create a new environment named `nlp` with python 3.7

```sh
conda create --name nlp python=3.7
conda activate nlp
```

The environment name (`nlp`) can be arbitrary, but you need to activate it every time you wish to run this experiment.

## Step 1. Install AllenNLP

Assuming you have already activated the environment, run

```sh
pip install allennlp
```

## Step 2. Install Other Packages with conda

```sh
conda install -c anaconda docopt
pip install pytrec_eval
pip install empath
pip install py-readability-metrics
python -m nltk.downloader punkt
```

Most of the libraries above are not required to run the demo experiment, simply comment out the corresponding `import` in `src` directory if you run into installation problems.

## Step 3. Prepare Data into the Right Format

The data is in json line format, meaning each line in the file is a json representing an individual's full posting history.

### Pre-training Data

For the pre-training step, the data locates at:

```
train: /fs/clip-psych/shing/umd_reddit_suicidewatch_dataset_v2/crowd/train/postprocess_posts_full_train.jsonl
dev: /fs/clip-psych/shing/umd_reddit_suicidewatch_dataset_v2/crowd/train/postprocess_posts_full_dev.jsonl
test: /fs/clip-psych/shing/umd_reddit_suicidewatch_dataset_v2/crowd/test/postprocess_posts_test_full.jsonl
```

Here is an example to demonstrate the file format (in the dataset, this json will be flatten into a single line):

```json
{
    "user_id": 849302,
    "label": "control",
    "tokens": [
                [
                    ["This", "is", "the", "first", "sentence"],
                    ["the", "second", "sentence"]
                ],
                [
                    ["The", "first", "sentence", "of", "the", "second", "document"],
                    ["2nd", "sentence", "of", "the", "2nd", "doc"],
                    ["third"]
                ]
              ],
    "subreddit": ["video", "funny"],
    "timestamp": [1376425356, 1391809118]
}
```

`label` is either `control` or `positive`, depending on whether the individual posted on SuicideWatch.

`tokens` is a `List[List[List[str]]]` field, representing the hierarchical structure of individual having multiple document, where each document can have multiple sentences, and each sentence can have multiple words.

`subreddit` is the subreddit forum the corresponding document is from, not used in training.

`timestamp` is the timestamp of the corresponding document, not used in training.

### Training Data

For the model training step, the data is located at:

```
train: /fs/clip-psych/shing/umd_reddit_suicidewatch_dataset_v2/crowd/train/task_A.train
dev: /fs/clip-psych/shing/umd_reddit_suicidewatch_dataset_v2/crowd/test/task_A.test
test: /fs/clip-psych/shing/umd_reddit_suicidewatch_dataset_v2/expert/cleaned_task_A.expert
```

The file format is identical to the pretraining format, except in this case, all subreddit will be from SuicideWatch (since this is task A), and the labels is either "a", "b", "c", or "d" (No, Low, Moderate, Severe).

## Step 4. Pretraining on the Weak Supervision Dataset

*Optional: skip step 4 and 5 if you have access to a trained model*

In the commend line (on a GPU-enabled machine, with `nlp` environment activated), type the following (but change `PRETRAIN_MODEL_PATH` and your path to the dataset accordingly)

```sh
cd PATH/TO/learning2assess/..
export RANDOM_SEED=$RANDOM
export PRETRAIN_TRAIN_DATA_PATH="/fs/clip-psych/shing/umd_reddit_suicidewatch_dataset_v2/crowd/train/postprocess_posts_full_train.jsonl"
export PRETRAIN_DEV_DATA_PATH="/fs/clip-psych/shing/umd_reddit_suicidewatch_dataset_v2/crowd/train/postprocess_posts_full_dev.jsonl"
export PRETRAIN_MODEL_PATH="YOUR/PATH/TO/STORE/PRETRAIN/MODEL/"
echo "training pretrain model"
allennlp train -f --include-package learning2assess -s $PRETRAIN_MODEL_PATH learning2assess/configs/pretrain_clpsych.json
```

## Step 5. Model Training on the Crowdsource Dataset

In the commend line (on a GPU-enabled machine, with `nlp` environment activated), type the following (but change `MODEL_PATH` accordingly)

```sh
cd PATH/TO/learning2assess/..
export MODEL_PATH="YOUR/PATH/TO/STORE/MODEL/"
export TRAIN_DATA_PATH="/fs/clip-psych/shing/umd_reddit_suicidewatch_dataset_v2/crowd/train/task_A.train"
export DEV_DATA_PATH="/fs/clip-psych/shing/umd_reddit_suicidewatch_dataset_v2/crowd/test/task_A.test"
echo "training tuned model"
allennlp train -f --include-package learning2assess -s $MODEL_PATH learning2assess/configs/tune_A_clpsych.json
```

After you finished training, you can go to the `MODEL_PATH` directory to see model performance on train and dev set. You can also visualize MODEL_PATH with `tensorboard`.

## Step 6. Generating Prediction

After you finished training (or have access to a trained model), you can do inference on the test data:

```sh
cd PATH/TO/learning2assess/..
export TEST_DATA="/fs/clip-psych/shing/umd_reddit_suicidewatch_dataset_v2/expert/cleaned_task_A.expert"
allennlp predict ${MODEL_PATH}/model.tar.gz $TEST_DATA --include-package learning2assess --predictor han_clpsych_predictor --output-file task_A.expert.prediction
```

If you don't have a GPU-enabled machine, add `--cuda-device=-1` to `allennlp predict`. See `allennlp predict -h` for all options.

This will output `task_A.expert.prediction` in json line format.
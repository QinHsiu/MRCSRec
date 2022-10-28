# MRCSRec

This is our Pytorch implementation for the paper: "**Meta-semantic Regularity for Contrastive Sequencial Recommendation**".

## Environment  Requirement

* Pytorch>=1.7.0
* Python>=3.7 

## Usage

Please run the following command to install all the requirements:  

```python
pip install -r requirements.txt
```

## Datasets Prepare

Please use the `data_process.py` under `dataset/` to  get the input dataset by running the following command :

```python
python data_process.py
```

## Evaluate Model

We provide the trained models on Amazon_Beauty, Amazon_Sports_and_Outdoors, and Yelp datasets in `./log/Checkpoint/<Data_name>`folder. You can directly evaluate the trained models on test set by running:

```
python run_seq.py --dataset=<Data_name> --do_eval
```

On Amazon_Beauty:

```python
python run_seq.py --dataset=Amazon_Beauty --do_eval
```

```
INFO  test result: {'recall@5': 0.0572, 'recall@10': 0.0884, 'recall@20': 0.1233, 'ndcg@5': 0.0357, 'ndcg@10': 0.0457, 'ndcg@20': 0.0545}
```

On Amazon_Sports_and_Outdoors:

```python
python run_seq.py --dataset=Amazon_Sports_and_Outdoors --do_eval
```

```
INFO  test result: {'recall@5': 0.0322, 'recall@10': 0.049, 'recall@20': 0.0728, 'ndcg@5': 0.0202, 'ndcg@10': 0.0256, 'ndcg@20': 0.0316}
```

On Yelp:

```python
python run_seq.py --dataset=Yelp --do_eval
```

```
INFO  test result: {'recall@5': 0.0443, 'recall@10': 0.0639, 'recall@20': 0.0945, 'ndcg@5': 0.0326, 'ndcg@10': 0.039, 'ndcg@20': 0.0467}
```

## Train Model

Please train the model using the Python script `run_seq.py`.

​	You can run the following command to train the model on Beauty datasets:

```
python run_seq.py --dataset=Amazon_Beauty --epochs=100 --use_regular=1 --joint=0 --lmd=0.1 --beta=0.1
```


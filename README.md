# FinCausal
## Task 1 
### Method 1 (BERT)
#### Training 
```
python train.py -c config.json -d 0
```

#### Testing
```
python test.py -r Fincausal_task1_method1.pth -d 0 --output submit.csv
```

### Method 2 (RoBERTa)
#### Training 
```
python train.py -c config.json -d 0
```

#### Testing
```
python test.py -r Fincausal_task1_method1.pth -d 0 --output submit.csv
```
## Task 2 
### Method 1 (BERT)
#### Training 
```
python train.py -c config.json -d 0
```

#### Testing
```
python test.py -r Fincausal_task2_method1.pth -d 0 --output test.csv
```


### Method 2 (CRF)
#### Training 
```
python train.py
```

#### Testing
```
python test.py
```



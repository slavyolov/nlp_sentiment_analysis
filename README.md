# macOS setup

### install pyenv (optional)

```buildoutcfg
brew install pyenv
echo 'eval "$(pyenv init --path)"' >> ~/.zprofile
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
```

### Python Virtual Environment

- create virtual environment (https://github.com/pyenv/pyenv)
```buildoutcfg
python -m venv ~/venvs/nlp-sa
```

- install project packages
```buildoutcfg
pip install --upgrade pip
pip install -r requirements.txt
```
- add "src" directory as source directory (PyCharm)

# Run the process
1. ```main.py``` takes care for : 
   - data_processing
   - EDA
   - sentiment_analysis algorithms
   - random sampling
   - combining the annotations
   - storing output files

2. ```main_evaluation.py``` takes care for :
   - calculating the model scores and performance assesment

# Details about the annotation
As a team we decided to do cross annotation. 

Every expert was responsible to label data subset and to pick one of three options (neutral, positive and negative).
When this stage was done, the labels were merged and with majority voting we selected as ground_truth label the most 
popular one for every row. These labels were later used to calculate parformance metrics

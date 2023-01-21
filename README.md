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






# Windows 10 setup

# Setup conda environment
- https://docs.anaconda.com/anaconda/user-guide/tasks/pycharm/#configuring-a-conda-environment-in-pycharm

# add gitignore (optional)
- https://stackoverflow.com/a/23970818
- https://github.com/github/gitignore/blob/main/Python.gitignore

# install the packages in the requirements.txt file

# Hints :
- reading files in Windows (replace "\\" with "/")
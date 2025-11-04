\
@echo off
python -m venv .venv
call .venv\Scripts\activate
pip install -r requirements.txt
python app.py train --models knn rf dt lr --test-size 0.2
python app.py evaluate

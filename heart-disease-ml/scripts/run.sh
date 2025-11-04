#!/usr/bin/env bash
set -e
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py train --models knn rf dt lr --test-size 0.2
python app.py evaluate

FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir pip setuptools wheel
COPY pyproject.toml README.md ./
COPY hiremindset ./hiremindset
COPY streamlit_app.py ./

RUN pip install --no-cache-dir -e .

ENV PYTHONUNBUFFERED=1

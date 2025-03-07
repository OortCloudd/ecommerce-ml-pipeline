@echo off
set AIRFLOW_HOME=%CD%\airflow
.venv-py310\Scripts\python -m airflow webserver

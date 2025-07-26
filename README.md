
### Read Me

### About

 processes operational data  when dealing with tickets queue and priority using AI to automate workflow.
 
 The idea is to pick 2 models to predict priority and what queue it is meant to be. According to the predicted priority a confidence has to be reached if that confidence is not reached then its flagged as skipped assuming that i will be delegated.

### Instructions

#### Pre Requests

Install uv package manager to operate the program

add .env in the `src/` folder

```.env
POSTGRES_USER=
POSTGRES_PASSWORD=
POSTGRES_DB=
POSTGRES_HOST=
POSTGRES_PORT=
```

*Forgot to add hugging face key to install dataset* 
#### Seed Dataset

The huggingface-hub key needs to be added to the project to download dataset.

To index the dataset the `uv run main.py -rsd` 

#### Configs

Picking models by changing the files in `src/config`

`src/config/data_config.yaml`
```yaml
rag_save_file_name: 'ingest_1'
test_factor: 0.05
```

`rag_save_file_name` is to point at the rag store wanted. 
`test_factor` is to split the data. (5% of the data eg)

`src/config/live_config.yaml`
```yaml
priority_model_name: 'MNLI'
queue_model_name: 'RAGLF'
priority_stop: False
confidence_on_low: 0.1
confidence_on_medium: 0.5
confidence_on_high: 0.95
```

`src/config/test_config.yaml`
```yaml
priority_model_name: 'MNLI'
queue_model_name: 'RAGLF'
test_name: 'rag_test'
confidence_on_low: 0.1
confidence_on_medium: 0.5
confidence_on_high: 0.95
```

`priority_model_name`/`queue_model_name` are used to select the model what wants to be used
`test_name` is to give a name for where the test results will be saved
`confidence_on_*` is to give a confidence tolerance to the level of priority
`priority_stop` if true will block relatively low confidence in live

#### Run ingestion (Pre training)

To ingest the dataset for the rag to work use `uv run main.py -ri` 
#### Running API

how to run uvicorn `uv run uvicorn api:app --reload`


<!--
TODO
Add the need of hugging face keys
talk about how spiting up the indexing for priority is good after fine tuning or using different models (pick the one with most certainty)


Add uv install and  usage as a ~ requirements.txt
TODO make an use requirements.txt docker method



Setup Instructions (README)
Database setup (Docker Compose for PostgreSQL preferred)
Migration commands (alembic upgrade head)
Server startup (uvicorn main:app --reload)
Test execution (pytest)
Environment variable configuration

Write about CLI


mention dask
-->
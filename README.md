# Heart-failure-prediction

## Problem Description
<p align="center">
    <!--img src="https://github.com/Romilagarwal/LLM-project/blob/main/image/problem.jpg" width=200 -->
   <img src=" https://github.com/Romilagarwal/heart-failure-prediction/blob/main/img/kenny-eliason-MEbT27ZrtdE-unsplash.jpg"width=200px>
</p>
<p align="center">
   <em>Image credits: <a href="https://unsplash.com/@neonbrand">Kenny Eliason on Unsplash</a></em>
</p>

> Cardiovascular diseases (CVD) are one of the leading cause of deaths worldwide. By examining the common risk factors, such as high blood cholestrol, chest pains, age risks and other factors featured in this dataset, a person's risk of having CVD could be detected earlier, thus reducing the number of deaths caused by CVD.

> Using a Machine Learning approach, this project's goal is to help researchers in identifying the importance and correlations of each of the risk factors mentioned above using existing CVD medical records. A Machine Learning classification model trained on this dataset could be used to predict if a new patient is potentially at risk of having heart disease.

## Dataset
* [Kaggle Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/)

## Dependency and environment management
The following steps are for reproducing the results of this repo on your local workstation.

### Reproducing this repo
* Create a new folder called `heart_failure_prediction` on your local workstation.
* In command prompt/terminal, navigate to this new folder. Then clone this repo to this new folder using `git clone https://github.com/Romilagarwal/heart-failure-prediction.git`.
* Alternatively you could use these options to clone this repo.
    * VSCode or
    * Github Desktop
* Once you have cloned this repo, create the virtual environment as in subsequent steps.

### [Optional] Install pip
* Usually pip is included if you have installed python. To check this, execute `pip --version` in command prompt/terminal. If you could see a version, pip is installed.
* If not, refer this installation guide: [Official pip installation guidelines](https://pip.pypa.io/en/stable/installation/).

### [Optional] Install pipenv
* `pip install pipenv`

### Create Virtual Environment using Pipenv
* If you have cloned this repo, `Pipfile` and `Pipfile.lock` should already be available in `heart_failure_prediction` and you can skip the following step.
* If not, navigate to `heart_failure_prediction` in your command prompt/terminal:
    * Execute `pipenv install numpy scikit-learn seaborn jupyter notebook xgboost streamlit flask requests gunicorn`.
    * If it's successful, you should see both `Pipfile` and `Pipfile.lock` inside this folder.

### Activate virtual env
* To activate the pipenv environment defined in `Pipfile` and `Pipfile.lock`, in command prompt/terminal:
    * Navigate to `heart_failure_prediction`.
    * Run `pipenv shell`.
* You can execute the following after `pipenv shell` to:
    * Open Jupyter Notebook: `jupyter notebook`.
    * Run `training.py` script to recreate the model: `python training.py`.
    * Start the heart prediction Streamlit app: see [Run Heart Prediction app on Streamlit locally](#run-heart-prediction-app-on-streamlit-locally).
    * Serve the heart prediction Flask API call: see [Serve app using Flask](#serve-app-using-flask).
    * Run the heart prediction Docker container: see [Containerization](#containerization).

## EDA
* A detailed EDA on the dataset is provided in Jupyter Notebook [part_1_preprocessing.ipynb]( https://github.com/Romilagarwal/heart-failure-prediction/blob/main/notebook/part_1_preprocessing.ipynb).

## Model training
* The best classifier model is picked by evaluating `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier` and `XGBClassifier` models.
* Evaluation metrics: Confusion Matrix and AUC Scoring.
* The final model is a finetuned `XGBClassifier`.
* Full explanation with visualization are found in Jupyter Notebook [part_2_modeling.ipynb](https://github.com/Romilagarwal/heart-failure-prediction/blob/main/notebook/part_2_modeling.ipynb).

## Exporting notebook to script
* The data cleaning, feature engineering, training and finetuning the final model are consolidated into script [training.py](https://github.com/Romilagarwal/heart-failure-prediction/blob/main/training.py).

## Model deployment
* Model is deployed with multiple options to allow running on local workstation, via a container, or internet:
    * Flask: able to serve directly (see [here](#serve-app-using-flask)), or via Docker (refer [Containerization](#containerization)).
    * Streamlit: runs locally (see [here](#run-heart-prediction-app-on-streamlit-locally)) and on Streamlit Cloud (URL link is provided at [Cloud Deployment](#cloud-deployment)).
* **Note**: Use any options as you wish. The **best option** would be access it online at [https://heart-vs.streamlit.app/](https://heart-vs.streamlit.app/).

### Serve app using Flask
* To serve the Flask prediction app:
    * In command prompt/terminal, navigate to `heart_failure_prediction` folder.
    * Execute `python predict_flask.py`.
    * If you see the message `* Serving Flask app 'heart_disease_app'`, the Flask app is ready for testing predictions.
* To try out predictions, execute `predict_flask_test.py`. You should see something like this:
```bash
{'hasHeartDisease': True, 'hasHeartDisease_probability': 0.7287337183952332}
Potentially at risk of heart disease. Follow-up examination recommended.
```
* To stop the Flask app, press CTRL+C in command prompt/terminal.

### Run Heart Prediction app on Streamlit locally
* In command prompt/terminal: `streamlit run heart_disease_prediction.py`.
* It should redirect you to a localhost page containing the Streamlit app automatically.

## Containerization
* The Flask app is also containerized using Docker.
* **Prerequisites**:
    * Install Docker, see [Docker Installation](https://docs.docker.com/engine/install/).
    * Make sure Docker service is up and running in local workstation.
* In command prompt/terminal, run `docker build -t heart-prediction-app .`
* Check that Docker image is created successfully:
    * `docker images`.
    * You should see `heart-prediction-app` in the list of Docker Repositories.
* Start a Docker container with the built image: `docker run -it -p 9696:9696 --rm --name heart_app heart-prediction-app:latest`. To check if the container is started:
    * In a new command prompt/terminal, execute `docker ps`.
    * You should see a running container with image `heart-prediction-app:latest` and name `heart_app`.
* To test the served Flask app, you can use the Python script `predict_flask_test.py`. Steps:
    * Run `docker exec -it heart_app bash`. This opens a bash terminal inside the running container.
    * Run `python predict_flask_test.py`.
    * You should see:
    ```bash
    {'hasHeartDisease': True, 'hasHeartDisease_probability': 0.7287337183952332}
    Potentially at risk of heart disease. Follow-up examination recommended.
    ```
    * Type `exit` to quite the bash terminal.
* To stop the running Docker container `heart_app`, execute `docker stop heart_app`.

## Cloud Deployment
* This prediction app is deployed to Streamlit Cloud, here's the url to try it: [https://heart-vs.streamlit.app/](https://heart-vs.streamlit.app/).
* The following steps are for those who are interested to replicate deployment to Streamlit Cloud:
    * Sign in to [Streamlit Cloud](https://streamlit.io/cloud) with Github account.
    * Follow the [instructions here](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app) to create a Streamlit app by specifying the Github repo, Streamlit app (for this repo, use [`heart_disease_prediction.py`](https://github.com/Romilagarwal/heart-failure-prediction/blob/main/heart_disease_prediction.py))  and provide a file for environment setup ([Pipfile](https://github.com/viviensiu/heart-failure-prediction/blob/main/Pipfile) was used here).


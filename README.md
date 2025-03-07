# Ecommerce ML Pipeline

Real-Time E-commerce ML Pipeline on AWS (using Windsurf IDE)
Introduction: This guide will walk you through building a real-time e-commerce recommendation pipeline on AWS from scratch, using the AI-powered Windsurf IDE for development. We’ll cover everything from setting up your project environment to deploying a live API. The instructions are beginner-friendly and organized into clear steps. By the end, you’ll have an end-to-end ML system that ingests data, trains models (using ALS for collaborative filtering and CatBoost for ranking), and serves real-time recommendations in production. Let’s get started!
1. Development Environment Setup
Set up your development environment and project structure to support the various stages of the MLOps pipeline.
Create a Project Folder Structure: In Windsurf IDE, create a new project directory (e.g., ecommerce-ml-pipeline). Organize subfolders for each component of the ML pipeline. For example:
bash
Copy
ecommerce-ml-pipeline/
├── README.md            # Top-level documentation for the project
├── requirements.txt     # Python dependencies (use `pip freeze > requirements.txt` to generate)
├── data/
│   ├── raw/             # Original dataset (input data)
│   └── processed/       # Processed data ready for modeling (feature engineered)
├── models/              # Trained model artifacts (saved CatBoost model, ALS matrices etc.)
├── notebooks/           # Jupyter notebooks for exploration or prototyping
├── airflow/
│   └── dags/            # Airflow DAG definitions for data pipeline orchestration
├── src/                 # Source code for the project
│   ├── data_processing/ # Scripts for data cleaning and preprocessing
│   ├── training/        # Scripts for model training (e.g., train_catboost.py, train_als.py)
│   ├── deployment/      # Code for the deployment (e.g., API server code)
│   └── monitoring/      # Scripts or utils for monitoring (e.g., metrics logging)
└── infrastructure/
    ├── terraform/       # Terraform IaC scripts for AWS resources (EKS cluster, S3, IAM, etc.)
    └── k8s/             # Kubernetes manifest files (Deployment, Service, etc.)
This layout separates concerns so you can manage data, models, deployment, and infra independently. For instance, raw data, interim processed data, and final datasets are kept under data/, and trained models (and any evaluation results) are stored under models/​
COOKIECUTTER-DATA-SCIENCE.DRIVENDATA.ORG
​
COOKIECUTTER-DATA-SCIENCE.DRIVENDATA.ORG
. The src directory contains modular code: you might have a config.py for shared settings, a dataset.py to download or load data, a features.py for feature engineering, and dedicated modules for training and inference​
COOKIECUTTER-DATA-SCIENCE.DRIVENDATA.ORG
.
Structure your code to separate data, model, and service logic. For example, you might write a function load_data() in your data module to fetch data from S3, a train_model() function in the training module to fit a model given training data, and an infer() function in the deployment module to generate recommendations using the trained models. Keeping code modular makes it easier to test components individually and reuse code.
Initialize Version Control: (Optional but recommended) Initialize a git repository in your project folder (git init) and make initial commits of your structure and code. This will be useful once we set up CI/CD (GitHub Actions) in a later step. You can use Windsurf’s integrated git features or the command line. Ensure you commit the README.md, code, and infrastructure scripts, but not sensitive credentials or large data files. This repo will later connect to GitHub for automation.
By the end of this setup, your Windsurf IDE workspace should reflect a well-organized ML project, ready to implement the pipeline.
2. Data Pipeline Implementation
Implement the data ingestion and preprocessing pipeline that feeds your machine learning models. We will use Amazon S3 for storage and Apache Airflow to orchestrate the data flow and processing tasks.
Upload the Dataset to S3: Obtain your e-commerce dataset (e.g., transaction history, user interactions, product catalog). This could be a CSV file, a collection of JSON logs, etc. Create an Amazon S3 bucket (via the AWS Console or CLI) to store the data, for example my-ecommerce-data-bucket. It’s good practice to enable versioning on this bucket so that you keep historical versions of data files​
DOCS.AWS.AMAZON.COM
. Upload the raw dataset to the S3 bucket (you can use the AWS CLI: for example, aws s3 cp local_data.csv s3://my-ecommerce-data-bucket/data/raw/). Note the S3 key where the data resides (e.g., s3://my-ecommerce-data-bucket/data/raw/ecommerce.csv) as our pipeline will read from this location.
Configure Apache Airflow: Use Airflow to define an ETL workflow that fetches and prepares data for modeling. If you prefer not to manage Airflow yourself, you can use AWS Managed Workflows for Apache Airflow (MWAA) which runs Airflow in AWS. Otherwise, you can run Airflow locally or on an EC2 instance for development. In either case:
Make sure Airflow has access to AWS credentials (if using MWAA, you’ll configure an IAM role; if local, you might use AWS access keys or attach an IAM role to the instance).
In the airflow/dags/ directory of your project, create a DAG file, e.g., ecommerce_pipeline_dag.py. This DAG will outline the sequence: data extraction, preprocessing, feature engineering, etc.
Define the DAG with appropriate schedule (e.g., daily or hourly depending on how real-time your pipeline needs to be). For now, you can set it to run on demand or daily.
Use Airflow operators to perform tasks:
S3 Sensor or Python Operator for Data Availability: The first task could wait for new data in S3 or simply start processing the existing raw data.
Data Preprocessing Task: A PythonOperator can call a function from your src/data_processing/preprocess.py script to clean the raw data. For large-scale data, you might instead use an EmrAddStepOperator to run a Spark job on EMR (more on this below).
Feature Engineering Task: Another PythonOperator can call your feature_engineering.py to generate model input features (or you can combine this with preprocessing in one task, depending on complexity).
Output Task: Save the processed/featured data back to S3 (e.g., as processed/train_dataset.csv for model training). You could also trigger the model training as part of the DAG (using a SageMaker Training Operator or simply kicking off the training script), but we will handle training in the next section.
Ensure each task is linked in order in the DAG (e.g., preprocess_task >> feature_eng_task >> notify_task).
For example, your Airflow DAG might: read input data from S3, perform data transformations, then output the cleaned data for model training​
FRANCISCOYIRA.COM
. Airflow will let you monitor and retry these steps easily. In Windsurf, you can use the Airflow plugin or simply edit the DAG file with AI assistance for code if needed.
Data Preprocessing (ETL): In the data preprocessing step (which you coded in preprocess.py and call via Airflow):
Load the raw data from S3. For small datasets, you can use pandas to read the CSV directly via S3 (using boto3 or AWS Data Wrangler). For large datasets that don’t fit in memory, use PySpark. PySpark can read from S3 (via Hadoop AWS connector) in a distributed manner.
Clean the data: remove or impute missing values, filter out irrelevant records (e.g., very inactive users or outlier transactions), and convert data types as needed.
If using PySpark on a cluster, you might launch an EMR cluster to run this at scale. Spark is designed for big data and can process large datasets in parallel across a cluster, whereas pandas runs on a single machine​
STACKOVERFLOW.COM
. For example, if your data is tens of millions of records, PySpark (on EMR) would be more suitable than pandas, which is constrained by a single machine’s memory and CPU​
STACKOVERFLOW.COM
​
STACKOVERFLOW.COM
. If data is small (fits in RAM), pandas is simpler and fine.
After cleaning, output the intermediate result (e.g., a cleaned transactions table) either to S3 (as a CSV/Parquet in the processed/ folder) or pass it to the next task in memory (if using one Airflow worker and moderate data size).
Example: If the dataset is user purchase history, you might remove invalid entries, parse timestamps into datetime, and combine data from multiple sources (joining product info into transaction records, etc.).
Feature Engineering for Models: After basic preprocessing, create features tailored for the two models:
ALS Model (Collaborative Filtering features): ALS (Alternating Least Squares) is used for collaborative filtering recommendations (finding latent user-item factors). The primary input is typically a user-item interaction matrix or a list of (user, item, rating/count) tuples. Feature engineering here means preparing that interaction data:
Map user IDs and item IDs to numeric indices (since ALS in Spark expects integer IDs for matrix factorization).
If implicit feedback (e.g., views or purchases without explicit ratings), decide on a weighting (for example, assign a default rating of 1 for a purchase event or use number of purchases as a “rating”). You might use an implicit-feedback ALS which is common for recommender systems.
Possibly filter out users or items with too few interactions (to reduce noise).
The output of this step could be a sparse matrix or just the triples of (user_index, item_index, value) ready for model training.
CatBoost Model (Ranking features): CatBoost is a gradient boosting decision tree algorithm that can handle categorical features natively​
DOCS.AWS.AMAZON.COM
. We will use it to re-rank or predict the probability of a user liking a given item. Feature engineering for this model might include:
User features: e.g., user demographics (age, location), or behavioral features (total number of purchases, average spend, product categories frequently bought).
Item features: e.g., product category, price, brand, popularity score (like overall rating or sales rank).
Interaction features: features specific to the user-item pair, e.g., how many times the user viewed this item, whether the user’s last purchase was this category, time since user’s last purchase, etc.
Labels for training: You’ll need a target variable for supervised learning. For a recommender, a common approach is to treat “whether the user purchased the item” as a binary label. To train the CatBoost model, you would create a training dataset of many (user, item, features...) examples with label = 1 if the user bought the item and 0 if not. You can use the ALS model’s output to generate negative samples: for each user, take some items ALS recommends that the user didn’t actually purchase and label those 0, while the items the user did purchase are label 1. This way, CatBoost learns to distinguish items the user will like among a candidacy list.
CatBoost’s handling of categorical variables means you can feed categories (like user location or item category) directly as features without one-hot encoding​
DOCS.AWS.AMAZON.COM
. You just need to specify which columns are categorical when training the model.
Perform these feature computations using either pandas (if data is manageable) or PySpark (if data is huge). For example, you might use pandas to merge the user feature table and item feature table into the interaction records​
REKKOBOOK.COM
​
REKKOBOOK.COM
. If data is in a SQL database or Spark, you could also compute features with SQL queries or Spark DataFrame operations.
Output of Feature Engineering: Save the final prepared datasets for model training. For instance:
s3://my-ecommerce-data-bucket/data/processed/als_input.csv containing user-item interaction triples for ALS.
s3://my-ecommerce-data-bucket/data/processed/catboost_train.csv and .../catboost_test.csv for CatBoost (you can split into training and validation sets).
You might also save mappings like user_id_to_index.json if needed for the ALS model to interpret results later.
Review the Data Pipeline DAG: Make sure your Airflow DAG ties these steps together. For example, the DAG might have tasks: wait_for_data -> preprocess -> feature_engineering -> finish. Test the DAG on a small sample of data to ensure each task runs. You can manually trigger the DAG in Airflow’s UI or via command. Monitor the logs in Airflow to debug issues. The goal is that with a single trigger, the pipeline:
Reads the raw data from S3,
Outputs cleaned data and features to S3,
(Optionally) triggers model training or notifies that data is ready for training.
At this point, you have an automated ETL pipeline that prepares fresh data for your ML models. In a production scenario, this DAG could be scheduled to run whenever new data arrives (e.g., daily at midnight or via an event trigger). The processed data in S3 will feed into the model training step next.
3. Model Training
With the data prepared, the next step is to train the recommendation models. We have two models in our architecture: an ALS collaborative filtering model and a CatBoost ranking model. We will use AWS managed services to scale out training: Amazon SageMaker for the CatBoost model (or AWS Batch/EC2 if preferred) and AWS EMR for the Spark-based ALS model. After training, we’ll save and version the models in S3.
Implement CatBoost Training Script: In src/training/train_catboost.py, write the code to train the CatBoost model. This script will:
Load the training data from the processed dataset (from S3). You can use boto3 to download the CSV to local storage on the training instance, or if using SageMaker, you will point SageMaker to the S3 location.
Use the CatBoost library to configure a model. For example:
python
Copy
import pandas as pd
from catboost import CatBoostClassifier, Pool
df = pd.read_csv('train_data.csv')
X = df.drop('label', axis=1)
y = df['label']
# Identify categorical feature indices or names:
cat_features = [0, 1, 2]  # (for example purposes, the indices of categorical cols in X)
train_pool = Pool(X, y, cat_features=cat_features)
model = CatBoostClassifier(iterations=1000, depth=6, learning_rate=0.1, eval_metric='AUC', early_stopping_rounds=20)
model.fit(train_pool, eval_set=... )
model.save_model('catboost_model.cbm')
Adjust parameters for your data (CatBoost can also auto-handle categorical features if you pass their indices). If the dataset is large, ensure you have enough memory; CatBoost training can be memory intensive but can run on CPU (CatBoost supports GPU too, but SageMaker’s CatBoost integration is CPU-focused​
DOCS.AWS.AMAZON.COM
).
The script should save the trained model artifact, e.g., to a file catboost_model.cbm (CatBoost’s binary model format) or even as a pickle if necessary. But using CatBoost’s own save is preferable.
Optionally, evaluate the model on a validation set and print metrics (AUC, accuracy, etc.) to logs so you know how it performed.
Implement ALS Training Script: In src/training/train_als.py, write code to train the ALS model. There are two ways to train ALS:
Using PySpark (Spark’s MLlib): This is suitable for large datasets. If using Spark, you will likely run this on an EMR cluster. The code would look like:
python
Copy
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
spark = SparkSession.builder.appName("TrainALS").getOrCreate()
interactions = spark.read.csv("s3://my-bucket/data/processed/als_input.csv", header=True, inferSchema=True)
# interactions schema: user_id, item_id, rating
als = ALS(maxIter=10, regParam=0.1, rank=50, userCol="user_id", itemCol="item_id", ratingCol="rating", coldStartStrategy="drop")
model = als.fit(interactions)
model.write().overwrite().save("s3://my-bucket/models/als_model")  # Saves ALS model in S3 (as a directory with Spark format)
You will need to ensure the Spark job has permission to write to that S3 path (via IAM roles).
Using implicit library (if data is moderate and can fit in one machine’s memory): The implicit Python library provides an ALS implementation for implicit feedback that runs on CPU using parallelism (not distributed but can handle reasonably large data). This approach might be easier to run on a single SageMaker instance or an EC2. Example:
python
Copy
import implicit
import scipy.sparse as sparse
# Assuming you loaded interactions into a sparse matrix (user x item)
als_model = implicit.als.AlternatingLeastSquares(factors=50, regularization=0.1, iterations=20)
als_model.fit(item_user_sparse_matrix)  # Note: implicit expects item-user matrix
als_model.save("als.model.npz")
If you use this, you’ll have to figure out how to load the model in the API (the library can load from the saved npz or you can save user/item factors).
For this guide, we’ll assume using Spark on EMR for scalability. Write the PySpark code as shown above.
Launch the ALS training on AWS EMR: You can use Airflow or the AWS CLI to submit this job. For example, use Airflow’s EmrAddStepsOperator to add a step that runs spark-submit with your train_als.py on a new EMR cluster. The EMR cluster can be transient (created for the job and auto-terminated). Alternatively, use AWS SageMaker Processing Jobs with a Spark container to run the Spark code (SageMaker has an option for Spark processing).
Ensure the ALS script saves the model. With Spark ML, saving the model to S3 as shown will output a directory with model data (which includes the factor matrices).
Run Training on SageMaker and EMR: Now that you have scripts, leverage AWS services to run them in a scalable way:
CatBoost on SageMaker: SageMaker can manage the training job on a dedicated machine cluster. You have two options:
Use SageMaker’s built-in CatBoost container. AWS has an integrated CatBoost algorithm in SageMaker​
DOCS.AWS.AMAZON.COM
. You would create a SageMaker Training Job specifying the built-in CatBoost image, your S3 input data location, and output model location. In AWS Console, this is a matter of selecting CatBoost, pointing to the S3 path of catboost_train.csv (and a validation file if you have one), and choosing instance type (e.g., ml.m5.2xlarge for CPU as recommended​
DOCS.AWS.AMAZON.COM
). SageMaker will automatically train and save the model to S3.
Or use a custom training script (the one you wrote). For this, you can either create a Docker image that runs train_catboost.py or use the SageMaker Python SDK from your environment to submit the script as a job. For example, using the SDK:
python
Copy
import sagemaker
from sagemaker.sklearn import SKLearn
estimator = SKLearn(entry_point='train_catboost.py', role='YourSageMakerRole',
                    instance_count=1, instance_type='ml.m5.xlarge',
                    framework_version='0.23-1')  # Example with sklearn container (or use ScriptProcessor)
estimator.fit({'train': 's3://my-bucket/data/processed/catboost_train.csv'})
However, using the built-in algorithm might be simpler for beginners.
ALS on EMR: Use the AWS CLI or Airflow to spin up the EMR job. For instance, using AWS CLI:
bash
Copy
aws emr create-cluster --name "ALS-Training" \
    --use-default-roles --release-label emr-6.5.0 --instances '[{"InstanceCount":3,"InstanceGroupType":"CORE","InstanceType":"m5.xlarge"},{"InstanceCount":1,"InstanceGroupType":"MASTER","InstanceType":"m5.xlarge"}]' \
    --steps '[{"Name":"ALS Train","ActionOnFailure":"TERMINATE_CLUSTER","HadoopJarStep":{"Jar":"command-runner.jar","Args":["spark-submit","s3://my-bucket/scripts/train_als.py"]}}]' \
    --auto-terminate
This command (as an example) creates a small cluster and runs the Spark submit of your script (assuming you uploaded train_als.py to S3 or you can bootstrap it). You can configure this in Terraform or through Airflow operators as well. The cluster will terminate after running.
Monitor Training Jobs: SageMaker training can be tracked in SageMaker Console (it shows logs and metrics), and EMR job progress can be seen in the EMR Console or through CloudWatch logs if configured. Wait for both jobs to complete successfully.
Store and Version Model Artifacts: Once training is done, you should have:
A CatBoost model file (on SageMaker, it will be saved to the S3 output path you specified; on local, you saved catboost_model.cbm which you should upload to S3 manually or via the SDK).
An ALS model (either a directory in S3 if using Spark ML, or a set of files if using another library).
Create a designated S3 path for models, for example s3://my-ecommerce-data-bucket/models/. You can organize by model name and date or version:
e.g., models/catboost/latest/catboost_model.cbm and models/als/latest/ for current production models.
Also keep a history: perhaps models/catboost/2023-12-01/model.cbm for versioning by date or a version number.
Since the S3 bucket has versioning enabled​
DOCS.AWS.AMAZON.COM
, if you overwrite the “latest” model file, AWS will keep older versions in case you need to roll back​
DOCS.AWS.AMAZON.COM
. (Alternatively, you could manually keep older files with different names.)
It’s a good idea to save additional metadata: for CatBoost, save the training metrics or feature importance; for ALS, you might save the mapping of user IDs if needed. These can be stored in the same S3 location (e.g., a models/als/metadata.json).
Register the model versions somewhere. For a beginner setup, a simple approach is to maintain a text file or use the README to note which data and training run corresponds to which model file. In more advanced setups, you’d use a model registry or an ML metadata tracking tool (like MLflow or SageMaker Model Registry), but that’s optional.
(Optional) Validate Models: Before deploying, it’s wise to do a quick sanity check:
Load the CatBoost model (you can use CatBoost’s load_model) and test it on a few examples or the validation set to ensure it performs as expected.
Load the ALS model and maybe retrieve some recommendations for a test user to see if they look reasonable (e.g., check that it returns items the user has a history with or similar ones).
These checks can be done in a Jupyter notebook or a small test script. They give you confidence that the pipeline produced a working model.
By the end of this stage, you have two trained models stored in S3 and ready to be served. Notably, this two-model approach (ALS for collaborative filtering + CatBoost for ranking) creates a powerful two-layer recommender system that can significantly improve recommendation quality over a single model​
GITHUB.COM
. Next, we’ll deploy these models behind a real-time API.
4. Deployment with Kubernetes and Terraform
Now we’ll deploy the inference service that uses the trained models to provide recommendations in real time. We’ll use Terraform to provision AWS infrastructure, including an Amazon EKS (Kubernetes) cluster to host our service, and Amazon ECR for our Docker images. Then we’ll create Kubernetes manifests to run our recommendation API on the cluster. This approach ensures our deployment is reproducible and scalable.
Provision AWS Infrastructure with Terraform: Using Infrastructure-as-Code (IaC) allows you to automate and track changes to your cloud setup​
DEV.TO
. In the infrastructure/terraform/ directory of your project, create Terraform configuration files (e.g., main.tf, vars.tf):
VPC and Networking: Define a VPC, subnets, and any security groups needed. (AWS EKS can use the default VPC or a custom one. Terraform AWS VPC module can simplify this.)
S3 Bucket: If you haven’t created the S3 bucket for data/models manually, define it in Terraform (aws_s3_bucket resource) with versioning enabled (aws_s3_bucket_versioning resource) so that the bucket is set up as needed.
ECR Repository: Define an aws_ecr_repository resource for your Docker images (e.g., repository name “ecommerce-recommender”).
EKS Cluster: Use the Terraform AWS EKS module or resources:
aws_eks_cluster for the control plane.
aws_eks_node_group (or launch template + aws_autoscaling_group) for worker nodes. Specify instance types (e.g., m5.large) and desired scaling settings (maybe start with 2–3 nodes).
Ensure the node group has an IAM role with permissions to access necessary AWS resources (S3, CloudWatch). You might attach a policy like AmazonS3ReadOnlyAccess to the node role so pods can read S3 models (or use IAM Roles for Service Accounts later).
IAM Roles: Terraform should create roles for the EKS cluster and nodes. For example, an AmazonEKSClusterRole and AmazonEKSWorkerNodeRole with appropriate AWS-managed policies. These roles are crucial for EKS operations​
DEV.TO
.
You can organize these into modules for clarity (VPC module, EKS module, etc.), which enhances reusability​
DEV.TO
.
After writing the configs, run Terraform commands to deploy:
bash
Copy
terraform init        # initialize Terraform and download AWS provider
terraform plan        # preview the resources to be created
terraform apply       # create the resources on AWS
Approve the plan when prompted. Terraform will then provision the EKS cluster, node group, ECR repo, etc. (This may take a few minutes for EKS control plane to be ready.)
Once done, use AWS CLI to get the Kubernetes config:
bash
Copy
aws eks --region <your-region> update-kubeconfig --name <cluster_name>
This command writes the cluster credentials to your ~/.kube/config so you can use kubectl. For example, aws eks --region us-east-1 update-kubeconfig --name my-eks-cluster​
DEV.TO
. Verify access with kubectl get nodes to see that your worker nodes are connected​
DEV.TO
.
Build the Recommendation API Docker Image: We will containerize the inference service (the code in src/deployment/).
Write the API Code: In src/deployment/app.py, implement a simple web server (using Flask or FastAPI) that loads the trained models and exposes an endpoint to get recommendations. For example, using FastAPI:
python
Copy
from fastapi import FastAPI
import boto3, pickle, catboost
from implicit.als import AlternatingLeastSquares

app = FastAPI()

# Load models at startup
s3 = boto3.client('s3')
# Download CatBoost model from S3
s3.download_file('my-ecommerce-data-bucket', 'models/catboost/latest/model.cbm', '/tmp/model.cbm')
cat_model = catboost.CatBoostClassifier()
cat_model.load_model('/tmp/model.cbm')
# Download ALS model (if small, or load factors if using implicit)
s3.download_file('my-ecommerce-data-bucket', 'models/als/latest/model.npz', '/tmp/als.npz')
als_model = AlternatingLeastSquares()
als_model.load('/tmp/als.npz')

@app.get("/recommendations")
def get_recommendations(user_id: int, top_k: int = 10):
    # Generate recommendations using ALS
    user_recs = als_model.recommend(user_id, filter_already_liked_items=False, N=top_k)
    # user_recs might be a list of (item_id, score) from ALS
    # Now use CatBoost to rerank or filter these
    features = []  # construct features for each (user, item) pair in user_recs
    # ... (feature engineering similar to training)
    cat_preds = cat_model.predict_proba(features)[:, 1]  # probability of like
    # Combine ALS scores and CatBoost predictions to rank final results
    final_recs = sorted(zip([item for item,score in user_recs], cat_preds), key=lambda x: x[1], reverse=True)
    return {"user_id": user_id, "recommended_items": [item for item,score in final_recs][:top_k]}
This is a simplified example. You’ll need to adapt it to however you combined ALS and CatBoost (some designs use ALS to generate candidate items and CatBoost to score them, as shown). Ensure the app can handle multiple requests efficiently (loading models in memory once at startup).
Create a Dockerfile: In your project root (or infrastructure/), write a Dockerfile for the API:
Dockerfile
Copy
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt  # This should include fastapi, uvicorn, boto3, catboost, implicit, etc.
COPY src/deployment/ ./deployment/
ENV PYTHONPATH=/app  # include /app in PYTHONPATH
CMD ["uvicorn", "deployment.app:app", "--host", "0.0.0.0", "--port", "80"]
If your model files are small, you could COPY them into the image (for example, if you had saved them in the repo). But more flexibly, as above, the app downloads the latest model from S3 on startup. This way, you don’t need to rebuild the image for each new model; you deploy the same app and it fetches current model. (Ensure the container’s IAM role or AWS credentials allow S3 read. On EKS, we gave node IAM S3 read access, which by default is used by pods.)
Build the Docker Image: From the project directory, build the image and tag it for ECR. For example:
bash
Copy
docker build -t recommender-api:latest .
Once built, tag it with your ECR repository URI:
bash
Copy
docker tag recommender-api:latest <aws_account_id>.dkr.ecr.<region>.amazonaws.com/ecommerce-recommender:latest
Push Image to ECR: Authenticate Docker to ECR and push the image. Use AWS CLI to get login credentials and pipe to Docker:
bash
Copy
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <acct_id>.dkr.ecr.<region>.amazonaws.com
(This logs you in to ECR from Docker, using an auth token)​
DOCS.AWS.AMAZON.COM
. Now push:
bash
Copy
docker push <aws_account_id>.dkr.ecr.<region>.amazonaws.com/ecommerce-recommender:latest
This uploads your image to the ECR repository​
DOCS.AWS.AMAZON.COM
. You can verify on the AWS ECR console that the image and tag appear.
Deploy to Kubernetes (EKS): With the image in ECR and EKS cluster running, define Kubernetes manifests to run the API.
Kubernetes Deployment: Create a file infrastructure/k8s/deployment.yaml:
yaml
Copy
apiVersion: apps/v1
kind: Deployment
metadata:
  name: recommender-api-deployment
  labels:
    app: recommender-api
spec:
  replicas: 3  # start with 3 pods for high availability
  selector:
    matchLabels:
      app: recommender-api
  template:
    metadata:
      labels:
        app: recommender-api
    spec:
      containers:
      - name: recommender-api
        image: <aws_account_id>.dkr.ecr.<region>.amazonaws.com/ecommerce-recommender:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 80
        env:
        - name: AWS_REGION
          value: "<your-region>"
        # If using IAM Roles for Service Accounts, you'd specify serviceAccountName here with proper annotation
This defines 3 replicas of our API Pod. We include an AWS_REGION env var so that boto3 knows which region to connect to. The container will pull the latest image from ECR (make sure your cluster can access ECR — by default, ECR is public to AWS accounts in the same region, and nodes can pull if they have the ECR read permission which is usually included in the EKS node role).
Kubernetes Service: Create infrastructure/k8s/service.yaml:
yaml
Copy
apiVersion: v1
kind: Service
metadata:
  name: recommender-api-service
spec:
  type: LoadBalancer
  selector:
    app: recommender-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 80
This exposes the deployment via an AWS ELB. Kubernetes will provision an ELB in your AWS account for this Service (because we chose LoadBalancer type). The service listens on port 80 and routes to the pods on port 80.
Apply Manifests: Use kubectl to create the resources:
bash
Copy
kubectl apply -f infrastructure/k8s/deployment.yaml   # deploys the pods
kubectl apply -f infrastructure/k8s/service.yaml      # creates the LoadBalancer service
Running kubectl apply will instruct EKS to launch the Deployment (which in turn creates the ReplicaSet and Pods)​
DEV.TO
. Then the Service will create an ELB and link it to those pods​
DEV.TO
. You can run kubectl get pods to watch the pods come up, and kubectl get svc to get the Service details. The Service will show an external EXTERNAL-IP (the DNS or IP of the ELB) once provisioning is complete.
Configure IAM for S3 Access: If your pods cannot access S3 due to permission, you have a couple of options:
Easiest: Edit the node instance role to have S3 read permissions to your bucket (if not already). Then pods will inherit that ability (on EKS, pods running on EC2 nodes by default use the node’s IAM role).
Better practice: Use IAM Roles for Service Accounts (IRSA) to give just the recommender pods access to S3. This involves creating an IAM role, annotating a Kubernetes ServiceAccount, and updating the Deployment to use that serviceAccountName. For a beginner setup, the node role approach is okay, but for production, consider IRSA to avoid over-permission of nodes.
DNS or Domain Setup: The LoadBalancer will provide a URL endpoint. You can map a custom domain via Route53 if desired, but for now you can use the AWS-provided DNS. (Find it in kubectl get svc output or AWS EC2 Console under Load Balancers).
At this point, the API is live. Test it (we will formally test in final steps, but you can do a quick curl now).
Infrastructure Verification: Use kubectl get all to verify all resources (pods, services, deployments) are running. Check ECR to ensure the image is there, and EKS that pods pulled the image successfully. If any pod is in CrashLoopBackOff, check its logs (kubectl logs <pod-name>) – you might need to adjust configurations or ensure the model files are accessible. Common fixes include setting AWS credentials or IAM correctly, or adjusting memory/CPU requests in Deployment if the container needs more resources. You might also set up an auto scaler later, but for now we fixed 3 replicas.
At this stage, you have a running Kubernetes cluster on AWS with your ML inference service deployed. The API uses the latest models from S3 to serve real-time recommendations.
5. CI/CD Pipeline
To streamline development and maintenance, set up a CI/CD pipeline using GitHub Actions. This will automate testing, building, and deploying your pipeline components whenever you update the code or data. We’ll also outline how to enable continuous training and redeployment of models as new data comes in (continuous learning).
GitHub Repository and Actions Setup: If you haven’t already, create a GitHub repository for your project and push the code (except large data files) to GitHub. In the repo, configure GitHub Actions by adding a workflow file (e.g., .github/workflows/main.yml):
Continuous Integration (CI) – Testing: Define a job that runs on each pull request or push. This job can use a Python environment to run any automated tests you’ve written. For instance, you might have unit tests for your data processing functions or a small integration test for the recommendation logic. Use the actions/setup-python action to set up Python, install dependencies, and run pytest. While writing tests might be beyond a beginner scope, ensure at least that the code compiles and basic functions run (maybe a dry run of train_catboost.py on a tiny sample).
Continuous Deployment (CD) – Build & Deploy: Define another job that triggers on push to main or when you create a new release. This job will:
Check out the repo code using actions/checkout.
Set up AWS credentials. Store your AWS Access Key, Secret, and region as GitHub Secrets (e.g., AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION). Use the official AWS action aws-actions/configure-aws-credentials@v1 to export these for use in the job.
Log in to ECR: Use AWS CLI (as shown earlier) or the action aws-actions/amazon-ecr-login@v1 to authenticate Docker to ECR.
Build Docker image: You can reuse the Dockerfile to build the image in the workflow. For example, run docker build -t $ECR_REPO:$GITHUB_SHA . (tag with commit SHA or use latest).
Push Docker image to ECR: Run docker push ... to upload the new image. You might tag it with the commit or a version. If tagging as latest, it will update the image that EKS uses.
Deploy to EKS: After pushing the image, update the Kubernetes deployment. You can use kubectl in the Actions workflow. First, set up kubectl (e.g., use azure/setup-kubectl action or install it via pip). Then use the AWS CLI to get kubeconfig (similar to the earlier command but you might need to ensure the IAM user has EKS access). Finally, run kubectl set image deployment/recommender-api-deployment recommender-api=<new-image> to update the image, or apply updated manifest files. Another approach is to have your Kubernetes manifests in the repo and apply them directly: kubectl apply -f infrastructure/k8s/deployment.yaml (if you updated the image tag inside it).
(Optional) Invalidate Cache or Restart Pods: If using image:latest, Kubernetes might not pull it unless the tag is updated with a new digest. You might force a rollout by adding an annotation or simply scaling down and up. Alternatively, tag images with commit hashes and update the deployment to that exact tag.
Workflow Triggers: Set the CI steps to run on every push to any branch (or PR), and the CD steps to run on merges to main or on manually triggered workflow dispatch. For model retraining (discussed next), you might have a schedule.
Example Workflow Outline​
EVERYTHINGDEVOPS.DEV
:
yaml
Copy
name: CI-CD Pipeline
on:
  push:
    branches: [ "main" ]
  pull_request:
    branches: [ "main" ]
  schedule:
    - cron: "0 0 * * 1"   # (for example, weekly retraining trigger)
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: "3.10"
      - run: pip install -r requirements.txt
      - run: pytest tests/   # if you have tests
  deploy:
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Configure AWS
        uses: aws-actions/configure-aws-credentials@v1
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ secrets.AWS_REGION }}
      - name: Login to ECR
        uses: aws-actions/amazon-ecr-login@v1
      - name: Build Docker Image
        run: docker build -t ${{ env.ECR_REPO }}:latest .
        env:
          ECR_REPO: "ecommerce-recommender"
      - name: Push to ECR
        run: docker push ${{ env.ECR_REPO }}:latest
        env:
          ECR_REPO: "<aws_account_id>.dkr.ecr.<region>.amazonaws.com/ecommerce-recommender"
      - name: Update K8s Deployment
        run: |
          aws eks update-kubeconfig --region ${{ secrets.AWS_REGION }} --name <cluster_name>
          kubectl set image deployment/recommender-api-deployment recommender-api=${{ env.ECR_REPO }}:latest
This is just an illustration; you may adjust according to your repo and naming. The key idea is that any code change triggers an automated build and deployment so you don’t manually rebuild containers each time.
Continuous Training & Deployment of Models: In an evolving e-commerce system, you might want to retrain models periodically as new data comes in (Continuous Training, CT). Here’s how you can incorporate that:
Scheduled Retraining: Use GitHub Actions scheduled trigger (cron) or rely on your Airflow. For example, you can set a weekly schedule in the Actions workflow (schedule: cron: ...)​
EVERYTHINGDEVOPS.DEV
 to run a job that triggers model retraining.
Automate Retraining: This job could execute a Python script in your repo that:
Loads new data or incremental data from S3 (or a database).
Runs the preprocessing and feature engineering (you could even invoke the Airflow DAG via its REST API or CLI from within this job, or directly call the functions if you can run them in GitHub Actions).
Submits a SageMaker training job for CatBoost (using AWS CLI or boto3) and waits for completion.
Submits an EMR step for ALS or uses AWS Glue/Athena for collaborative filtering updates.
When models are trained, it uploads the new model artifacts to S3.
Automate Redeployment: Once new models are in S3, you need the API to use them. If your API is coded to always fetch the “latest” model from S3 on each request or startup, you can simply restart the pods to load the new model. You could automate this by having the retraining job call kubectl rollout restart deployment/recommender-api-deployment or scale the deployment down and up, causing pods to reinitialize and fetch the new model. Alternatively, if you package models into the container, then the CI/CD pipeline would need to rebuild the image with the new model file – this is less dynamic. The design we took (loading from S3 at runtime) allows the running service to update models without rebuilding the container image.
Integration with Airflow: Another approach is to let Airflow handle the whole loop: use an Airflow DAG to retrain models and then maybe call a Kubernetes API or Argo CD hook to deploy the new model. But to keep it simple, using GitHub Actions for scheduled retraining is fine for now.
Continuous Delivery of Data Pipeline Code: If you update your data processing or training code, those changes will also flow through CI/CD. For example, if you optimize a feature in feature_engineering.py, you’d commit and push, then possibly trigger a retraining via the same pipeline.
Protect Secrets and Keys: In your CI/CD setup, ensure that sensitive information (AWS keys, database passwords if any) are stored in GitHub Secrets or Parameter Store, not in plaintext in the code. Windsurf IDE’s integration with GitHub should not expose these. When setting up GitHub Actions, you configured secrets​
EVERYTHINGDEVOPS.DEV
​
EVERYTHINGDEVOPS.DEV
, which is the correct approach.
With CI/CD in place, any change in your repository can be automatically tested and deployed. This reduces manual work and chance of error, and it allows you to continuously improve the pipeline. For instance, you can schedule nightly model retraining so the recommendations stay up-to-date with the latest user behavior. The automated pipeline will help maintain a robust, scalable, and efficient ML system​
AWS.PLAINENGLISH.IO
.
6. Monitoring & Logging
Once the system is running, it’s critical to monitor its performance and log important information. We will set up Prometheus and Grafana for monitoring the Kubernetes cluster and the API, and use AWS CloudWatch for logging and additional alarms. Monitoring will alert you to issues like degraded model performance or infrastructure failures so you can respond quickly.
Deploy Prometheus for Metrics Collection: Prometheus will collect metrics from your application and cluster. Many Kubernetes deployments use the Prometheus Operator or kube-prometheus stack for easy setup. You can do this in a number of ways:
Using Helm: Add the Prometheus community helm repo and install the stack:
bash
Copy
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install k8s-monitor prometheus-community/kube-prometheus-stack
This will install Prometheus, Alertmanager, and Grafana in your EKS cluster, along with default scraping configurations for Kubernetes components.
Manual Deployment: Alternatively, deploy Prometheus components with manifest files. For a beginner, Helm is simpler.
Ensure Prometheus is scraping your application metrics. If you used FastAPI or Flask, you can integrate something like Prometheus Python client to expose an endpoint (e.g., /metrics) with custom metrics (like request count, latency, recommendations made, etc.). Then you’d add a Pod annotation for Prometheus to scrape it or update the Prometheus config. If using the kube-prometheus-stack, it likely will scrape any pods annotated with prometheus.io/scrape: "true".
Prometheus will also gather cluster metrics (CPU, memory of pods, etc.) which is useful to ensure your service has enough resources.
Grafana installation often comes with the Prometheus stack above. Grafana will allow visualization of metrics via dashboards.
Set Up Grafana Dashboards: With Grafana (either the one from the Helm chart accessible via a LoadBalancer/NodePort or using AWS Managed Grafana), create dashboards to monitor:
API Performance: Dashboard showing request rate, error rate (e.g., HTTP 500 count), and latency (e.g., 95th percentile response time). If you instrumented your API with these metrics, you can graph them. If not, you can use the default Kubernetes NGINX ingress or service metrics (but in our simple setup, we might not have those).
Model Metrics: You might log a metric for “recommendation quality” if you have a way to measure it online (often hard in real-time without user feedback). At least monitor the distribution of model scores or something if possible.
Resource Utilization: Use Grafana to show CPU and memory usage of the recommender pods and the nodes. This helps in capacity planning (e.g., if the CPU is 90%+ you might need larger instances or more pods).
Grafana allows setting up alerts as well (which can send emails or Slack messages if thresholds are crossed).
Centralize Logging with CloudWatch: AWS CloudWatch can aggregate logs from various sources:
Application Logs: The logs from your API pods (stdout/stderr) can be shipped to CloudWatch. One way is to use Fluent Bit as a DaemonSet on the cluster to stream logs to CloudWatch Logs. AWS provides an Fluent Bit image and config for EKS. (There’s an AWS Distro for OpenTelemetry as well that can send logs to CloudWatch).
Airflow Logs: If using MWAA, the logs are already in CloudWatch by default. If running Airflow yourself, consider configuring it to log to CloudWatch (perhaps by using CloudWatch agent on the server).
SageMaker and EMR Logs: SageMaker training jobs output to CloudWatch Logs under /aws/sagemaker/TrainingJobs, so you can review past training runs. EMR can be set to log steps to CloudWatch as well.
By consolidating logs in CloudWatch, you can search them easily and set up metric filters. For example, you could create a CloudWatch Metric Filter for the API logs to count occurrences of the word “ERROR” and then alarm on it.
Also enable EKS control plane logging (API server, scheduler) to CloudWatch for completeness, through EKS settings or Terraform.
Set Up Alerts for Anomalies: Utilize both Prometheus Alertmanager and CloudWatch Alarms:
Alertmanager (Prometheus): If you installed the kube-prometheus-stack, Alertmanager is included. You can configure alert rules in Prometheus. For instance, create an alert if the API error rate > 5% for 5 minutes, or if no recommendations have been served for some time (could indicate a crash). You’d then configure Alertmanager to send notifications (email, Slack, etc.). Start with simple ones like high error rate or high latency.
CloudWatch Alarms: In the AWS CloudWatch console or via Terraform, set alarms on critical metrics:
CPUUtilization of EC2 nodes > 85% for 15 minutes -> could notify you to upscale.
If using an ELB from the Service, you can monitor the ELB’s RequestCount and HTTP 5xx errors with CloudWatch and alarm if errors spike.
Alarm on SageMaker job failures (SageMaker can emit a Failed metric or you can catch it via events).
Alarm on low disk space if relevant (for example, if the instance running Airflow is low on space).
Incident Response: Make sure these alarms notify you (via AWS SNS to email or PagerDuty, etc.). For a beginner setup, email is fine.
Model Performance Monitoring: Monitoring an ML system isn’t just about system metrics, but also data and model drift:
Log or monitor the input data characteristics. If the distribution of incoming product views/purchases changes drastically (say a feature that was usually 0/1 is now mostly 1, or new categories appear), you might need to retrain more often or investigate.
If possible, measure recommendation success. This could be user engagement with recommended items if you have feedback (but that’s more advanced and requires connecting the recomms served to outcomes).
You can store daily snapshots of certain stats (like average predicted score, etc.) and review them.
Remember that observability is a cornerstone of a well-architected system – AWS provides native services like CloudWatch and also supports open-source tools like Prometheus/Grafana for comprehensive monitoring and alerting​
EKSWORKSHOP.COM
. By combining these, you cover both infrastructure and application-level insights. In summary:
Prometheus/Grafana monitor detailed metrics of the API and Kubernetes cluster (helpful for real-time debugging and historical trends).
CloudWatch Logs collect all logs (useful for tracing errors, auditing, and any AWS service logs).
Alerts/Alarms ensure that when something goes wrong (model response slowed, service down, data pipeline failed, etc.), you get notified immediately to take action. This completes the loop of MLOps by adding continuous observability and reliability to the pipeline​
EKSWORKSHOP.COM
.
7. Finalizing the Project
At this point, you have a functional MLOps pipeline. The last step is to document everything and run end-to-end tests to ensure the system works as expected. You also want to make sure that the entire setup is reproducible and maintainable for future iterations.
Comprehensive Documentation: Create or update the README.md at the root of your project to serve as a guide for anyone using or inspecting the project. The README should include:
Project Overview: A summary of the pipeline’s purpose (e.g., “This project implements a real-time product recommendation system for e-commerce using an ALS model for candidate generation and a CatBoost model for ranking.”).
Architecture: Describe the components (maybe include a simple diagram if possible): data flow from S3 -> Airflow -> S3 -> model training (SageMaker/EMR) -> model in S3 -> API on EKS. You can list the AWS services and what they do.
Setup Instructions: How to set up the environment (essentially a short version of steps 1 and 2). This includes how to configure AWS credentials, how to deploy Terraform (with any prerequisites), and how to run Airflow if needed.
Usage: Explain how to trigger the pipeline. For example, “to retrain models, run the Airflow DAG or trigger the GitHub Actions workflow.” And “to query the recommender API, send a GET request to /recommendations?user_id={id} at the load balancer URL.”
Repository Structure: Explain the directory layout and where to find key files (you can copy the project tree structure as a reference)​
COOKIECUTTER-DATA-SCIENCE.DRIVENDATA.ORG
​
COOKIECUTTER-DATA-SCIENCE.DRIVENDATA.ORG
.
Team and License: If relevant, add author name(s) and an open-source license.
Notes: Any tips for future improvements, known limitations, etc. For instance, note that in a real production, one might use AWS Step Functions instead of Airflow, or use AWS CodePipeline instead of GitHub Actions, etc., but this project uses open-source and basic services for clarity.
Windsurf IDE users can also benefit from notes on how you utilized Windsurf’s features (like AI code completion) to speed up development, if you want to highlight that.
End-to-End Testing: Perform a full run of the pipeline on a small scale to validate each part works together:
Data Pipeline Test: Add a new sample data file to S3 or modify the existing one slightly, then manually trigger the Airflow DAG (via the Airflow UI or CLI). Monitor it to see that it finishes successfully and that new processed files appear in S3.
Model Training Test: You might not retrain completely due to time, but you could run the training scripts on a small subset just to ensure there are no errors. Alternatively, if using SageMaker, kick off a training job with a small epoch or two (or use a smaller dataset) to run quickly, just to test the integration.
Deployment Test: Make a request to the live API. You can use curl:
bash
Copy
curl http://<LoadBalancerDNS>/recommendations?user_id=123
Replace <LoadBalancerDNS> with the DNS name from kubectl get svc. You should get back a JSON response with a list of recommended item IDs. Check that the format is correct and that it returns quickly (a few hundred milliseconds ideally).
Test edge cases: try a user_id that doesn’t exist in the model (the API should handle it – maybe return an empty list or fallback popular items). Try different top_k parameters if exposed. If your API requires authentication (not in our simple design), test that too.
If possible, simulate a simple feedback loop: e.g., pretend user 123 bought item X that was recommended, then add that to a test dataset and retrain to see if model would reinforce that. This is more of a ML validation than pipeline, so optional.
Reproducibility Check: If someone new were to set this up, could they follow your instructions easily? Since you are effectively writing this guide as if for an LLM to follow along in Windsurf, it should be explicit enough. Ensure that:
The Terraform config in source control can recreate the AWS infra exactly (try destroying and re-applying if feasible).
The Airflow DAG and scripts are all in the repository and not relying on something only you have locally.
All configuration (like bucket names, cluster name, etc.) are either documented or parameterized (you might use Terraform variables or a config file in the code).
The environment (package versions) is captured in requirements.txt so that if someone else creates a new venv and installs, they get the right versions.
Randomness: If your model training involves randomness (e.g., random init in ALS), set a random seed for consistency in results during tests.
Data: Note that the actual dataset might be proprietary or large, so either mention where to get a public dataset or include a small dummy dataset for demonstration purposes.
Performance Considerations: As a final note in documentation or for the team, outline how the system can be scaled:
For example, if traffic increases, you can increase replicas or use Kubernetes HPA (Horizontal Pod Autoscaler) to scale the API based on CPU or request load.
If data volume increases, maybe use a larger EMR cluster or enable incremental training rather than full retraining.
Discuss any bottlenecks found during testing and possible mitigations (e.g., model loading takes time – could be improved by model server or optimizing model size).
Also, note costs: running an EKS cluster and EMR and SageMaker has costs; consider shutting down resources when not in use for development.
Final Review: Go through each part of the pipeline one more time and ensure all pieces are connected:
The Airflow DAG should be aware of where to put data for training.
The training job should output model artifacts where the API expects them.
The API should be configured to fetch the correct model files (check S3 paths, etc., maybe make them configurable via environment variables rather than hard-coded).
CI/CD secrets and config should be in place (one common mistake is forgetting to add the kubeconfig or AWS creds to GH Actions and then the deployment step fails – make sure those are set).
Once everything is verified, you will have a fully operational ML pipeline: from raw e-commerce data to real-time personalized recommendations in production. This pipeline includes automated data processing, model training, containerized deployment on Kubernetes, continuous integration/deployment, and robust monitoring. With this infrastructure, you can iteratively improve your recommendation algorithms (try new features, new model types, etc.) and have confidence that you can quickly deploy those improvements to users.Conclusion: Following this step-by-step guide, you’ve set up a real-time e-commerce ML pipeline on AWS using Windsurf IDE for development. You created a structured project, implemented data engineering with Airflow, trained two ML models (ALS & CatBoost) using scalable AWS services, deployed the models as an API on Kubernetes (EKS) with Terraform-managed infrastructure, and automated the workflow with CI/CD and monitoring. This modern MLOps approach ensures that your recommendation system is not only initially successful but also maintainable and extensible over time. Great job on reaching the finish line – your pipeline is live! 🚀
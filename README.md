# Smoking Behavior Prediction

This project is a comprehensive machine learning application that predicts whether a person is a smoker based on their demographic and socio-economic information. The system is architected as a modern, multi-service web application, fully containerized with Docker for easy deployment and scalability.

The front end is a dynamic, single-page application built with React that provides real-time predictions and model interpretability visualizations. It communicates with a dedicated Flask REST API that serves predictions from a trained logistic regression model.

![Application Screenshot](https://i.imgur.com/your-screenshot-url.png) 
*(**Action Required:** Take a screenshot of your final running application and replace the URL above.)*

## Features

* **Interactive UI:** A responsive, dashboard-style interface built with React for a seamless user experience.
* **Real-Time "What-If" Analysis:** Modify input values and see the prediction update instantly without a page reload.
* **Model Interpretability:** The UI displays the key factors influencing each prediction, separating them into those that increase and decrease risk.
* **Global Insights:** Shows the most influential features for the model's overall decision-making process.
* **Prediction Confidence:** Each prediction is accompanied by a confidence score (High, Medium, or Low).
* **Persistent History:** Prediction history is saved in the browser's `localStorage` for a continuous user experience.
* **Decoupled Architecture:** A dedicated Flask API serves the model, while a separate Flask web app serves the UI, proxied by Nginx.
* **Containerized Deployment:** The entire multi-service application is managed by Docker and Docker Compose for one-command startup.

## Tech Stack & Architecture

This project utilizes a modern, service-oriented architecture.

* **Machine Learning Pipeline (`main.py`):**
    * **Pandas:** For data loading and manipulation.
    * **Scikit-learn:** For building the preprocessing and modeling pipeline (`ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, `LogisticRegression`).
    * **Imbalanced-learn:** To handle class imbalance in the dataset using the `SMOTE` technique.

* **Backend (`api.py`):**
    * **Flask & Gunicorn:** A dedicated microservice to serve the trained model via a RESTful API.

* **Frontend (`app.py` & `templates/index.html`):**
    * **Flask & Gunicorn:** A lightweight web server responsible for rendering the main HTML page.
    * **React:** For building the dynamic and interactive user interface.
    * **Jinja2:** Used only to bootstrap the React component with initial data.

* **Infrastructure:**
    * **Docker & Docker Compose:** To containerize and orchestrate the multi-service application (`api`, `webapp`, `nginx`).
    * **Nginx:** As a reverse proxy to route incoming traffic to the appropriate backend service.

### Architecture Diagram
```
[ User Browser ] <--> [ Nginx (Port 80) ]
       |                      |
       |--- (/) ------------> [ Webapp Service (Flask/Gunicorn) ] --> Serves React UI
       |
       |--- (/predict) -----> [ API Service (Flask/Gunicorn) ] --> Returns JSON Prediction
       |
       |--- (/global) ------> [ API Service (Flask/Gunicorn) ] --> Returns JSON Insights
```

## Local Development Setup

The entire application is containerized, making setup incredibly simple.

### Prerequisites
* [Docker](https://www.docker.com/products/docker-desktop/) must be installed and running.

### Running the Application

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd smoking-behavior-prediction
    ```

2.  **Create the environment file:**
    Create a `.env` file in the root of the project and add a secret key:
    ```
    SECRET_KEY=a-very-strong-and-secret-key-that-you-make-up
    FLASK_DEBUG=true
    ```

3.  **Build and Run with Docker Compose:**
    This single command will build the Docker image (which includes training the model) and start all three services.
    ```bash
    docker-compose up --build
    ```

4.  **Access the Application:**
    Once the containers are running, open your web browser and navigate to:
    [http://localhost](http://localhost)

5.  **Stopping the Application:**
    To stop all services, press `Ctrl + C` in the terminal. To remove the containers and network, run:
    ```bash
    docker-compose down
    ```

## Future Improvements

* **Production-Ready Front-End Build:** Instead of using the in-browser Babel transformer, set up a Node.js build process (e.g., with Vite or Create React App) to pre-compile the React code for better performance.
* **CI/CD Pipeline:** Implement a GitHub Actions workflow to automatically run tests, build the Docker images, and push them to a container registry like Docker Hub.
* **Model Monitoring:** Add logging in the API to record incoming prediction requests and the model's output, which could be used to monitor for model drift over time.
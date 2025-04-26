# Bank Term Deposit Subscription Prediction

This project provides a machine learning-based solution for predicting whether a client will subscribe to a term deposit based on various client and campaign features. The model uses a pre-trained pipeline to make predictions based on user input, deployed using **Streamlit** for an interactive web application.

## Project Description

The goal of this project is to predict whether a client will subscribe to a term deposit, a type of fixed-term financial product. The model is trained using a real-world dataset, which includes features like:

- Client's demographic details (e.g., age, job, marital status)
- Campaign details (e.g., number of contacts during the campaign, previous contacts)
- Economic indicators (e.g., consumer price index, employment rate)

The model is built using a **Logistic Regression** classifier and is integrated with a Streamlit web application to provide an easy-to-use interface for users to input client information and receive predictions.

## Features

- **Client Input Form**: Users can input client details such as age, campaign data, and other relevant information.
- **Prediction Output**: The application predicts whether the client is likely to subscribe to a term deposit based on input data.
- **Metrics Display**: The application displays model evaluation metrics, such as accuracy score, confusion matrix, and classification report.
- **Interactive Visualizations**: The application provides a visual overview of the dataset and model performance, making it easier for users to understand the prediction logic.

## Installation

### Prerequisites

Before running the application, ensure you have the following dependencies installed:

- Python 3.6+
- Streamlit
- scikit-learn
- pandas
- joblib

### Installation Steps

1. Clone this repository:
   ```bash
   git clone https://github.com/zeeshan-akram-ds/Bank-Term-Deposit-Subscription-Prediction.git
2. Navigate into the project directory:
   cd Bank-Term-Deposit-Subscription-Prediction
3. Install the required Python packages:
   pip install -r requirements.txt
4. Run the Streamlit app:
   streamlit run app.py
This will start the web app and open it in your browser.
### Model Details

## Model Evaluation Metrics:

 # Accuracy: 0.82
 # Confusion Matrix:
    [[6113 1195]
    [ 318  610]]
 # Classification Report:
               precision    recall  f1-score   support

         0       0.95      0.84      0.89      7308
         1       0.34      0.66      0.45       928  
  accuracy                           0.82      8236    
 macro avg       0.64      0.75      0.67      8236  
 
weighted avg 0.88 0.82 0.84 8236


## Application Interface

The **Streamlit** application allows users to input the following details about a client:

### Client Information
- **Age**
- **Contacts during Campaign**
- **Days Since Last Contact**
- **Previous Contacts**
- **Employment Variation Rate**
- **Consumer Price Index**
- **Consumer Confidence Index**
- **Euribor 3 Month Rate**
- **Number of Employees**

### Client Profile
- **Job**
- **Marital Status**
- **Education Level**
- **Last Contact Month**
- **Last Contact Day**
- **Outcome of Previous Campaign**

After filling out the form, the user can click the "Predict Subscription" button to get the prediction along with the probability of the client subscribing to the term deposit.

## Acknowledgments

- **Dataset**: The dataset used in this project is based on the Kaggle "Bank Marketing" dataset.
- **Streamlit**: For providing an easy-to-use framework to deploy machine learning models as web applications.
- **scikit-learn**: For providing a wide range of tools for building and evaluating machine learning models.

## Contact

For any questions or feedback, please feel free to reach out via [zeeshanakram1704@gmail.com].

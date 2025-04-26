import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

pipeline = joblib.load('bank_pipeline.pkl')

# title
st.title("Bank Term Deposit Subscription Prediction")
st.write("Fill out the form below and let's predict if the client will subscribe to a term deposit!")

st.header("Client Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100)
    campaign = st.number_input("Contacts during Campaign", min_value=1, max_value=50)
    pdays = st.number_input("Days Since Last Contact", min_value=-1, max_value=1000)
    previous = st.number_input("Previous Contacts", min_value=0, max_value=50)
    emp_var_rate = st.number_input("Employment Variation Rate", min_value=-3.0, max_value=2.0, step=0.1)

with col2:
    cons_price_idx = st.number_input("Consumer Price Index", min_value=90.0, max_value=100.0, step=0.01)
    cons_conf_idx = st.number_input("Consumer Confidence Index", min_value=-60.0, max_value=0.0, step=0.1)
    euribor3m = st.number_input("Euribor 3 Month Rate", min_value=0.0, max_value=6.0, step=0.01)
    nr_employed = st.number_input("Number of Employees", min_value=4900.0, max_value=5300.0, step=0.1)

st.header("Client Profile")

col3, col4 = st.columns(2)

with col3:
    job = st.selectbox("Job", 
        options=['housemaid', 'services', 'admin', 'blue-collar', 'technician', 'retired',
                 'management', 'unemployed', 'self-employed', 'unknown', 'entrepreneur', 'student'])

    marital = st.selectbox("Marital Status", options=['married', 'single', 'divorced'])

    education = st.selectbox("Education Level",
        options=['basic_education', 'high_school', 'professional_course', 'unknown', 'university_degree'])

with col4:
    month = st.selectbox("Last Contact Month",
        options=['may', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec', 'mar', 'apr', 'sep'])

    day_of_week = st.selectbox("Last Contact Day", options=['mon', 'tue', 'wed', 'thu', 'fri'])

    poutcome = st.selectbox("Outcome of Previous Campaign", options=['nonexistent', 'failure', 'success'])

if st.button("Predict Subscription"):
    with st.spinner('Making prediction... Please wait'):
        # Prepare input
        input_df = pd.DataFrame({
            'age': [age],
            'campaign': [campaign],
            'pdays': [pdays],
            'previous': [previous],
            'emp.var.rate': [emp_var_rate],
            'cons.price.idx': [cons_price_idx],
            'cons.conf.idx': [cons_conf_idx],
            'euribor3m': [euribor3m],
            'nr.employed': [nr_employed],
            'job': [job],
            'marital': [marital],
            'education': [education],
            'month': [month],
            'day_of_week': [day_of_week],
            'poutcome': [poutcome]
        })

        prediction = pipeline.predict(input_df)[0]
        probability = pipeline.predict_proba(input_df)[0][1]

        st.success("Prediction complete!")

        if prediction == 1:
            st.success(f"The client is **likely to subscribe** to a term deposit. (Probability: {probability:.2%})")
        else:
            st.error(f"The client is **not likely to subscribe** to a term deposit. (Probability: {probability:.2%})")

with st.expander("Show Model Evaluation (click to expand)"):
    st.header("Model Performance on Test Data")

    # metrics
    accuracy = 0.8162943176299174
    st.metric(label="Accuracy", value=f"{accuracy:.2%}")

    # Confusion Matrix
    cm = [[6113, 1195],
          [318, 610]]

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Not Subscribed", "Subscribed"],
                yticklabels=["Not Subscribed", "Subscribed"],
                ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.subheader("Confusion Matrix")
    st.pyplot(fig)

    # Classification Report
    classification_rep = """
                  precision    recall  f1-score   support

        0       0.95      0.84      0.89      7308
        1       0.34      0.66      0.45       928

    accuracy                           0.82      8236
    macro avg       0.64      0.75      0.67      8236
    weighted avg    0.88      0.82      0.84      8236
    """

    st.subheader("Classification Report")
    st.code(classification_rep)

    st.header("Train vs Test Accuracy")
    train_acc = 0.8123254401942926
    test_acc = 0.8162943176299174

    fig2, ax2 = plt.subplots()
    labels = ['Train Accuracy', 'Test Accuracy']
    values = [train_acc, test_acc]
    colors = ['#4CAF50', '#2196F3']

    ax2.bar(labels, values, color=colors)
    ax2.set_ylim(0,1)
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Train vs Test Accuracy Comparison')

    for i, v in enumerate(values):
        ax2.text(i, v + 0.01, f"{v:.2%}", ha='center', fontweight='bold')

    st.pyplot(fig2)
st.markdown("""---""")
st.markdown(
    """
    <div style="text-align: center; padding: 10px; font-size: 14px; color: grey;">
        Built by <a href="https://github.com/zeeshan-akram-ds" target="_blank"><b>Zeeshan Akram</b></a> | Powered by Machine Learning and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
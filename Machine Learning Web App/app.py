import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import precision_score, recall_score 
import matplotlib.pyplot as plt

def main():
    st.title("Binary Classification Web App")
    st.write("Welcome to the binary classification web app!")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Is Mushroom edible or Poisonous🍄")
    st.sidebar.markdown("Is Mushroom edible or Poisonous🍄")


    @st.cache_data
    def load_data():
#        path = 'https://github.com/S-Mehran/Projects/blob/1cab01d7ae96064cc7cf298d9cf2b5e5ac296137/Machine%20Learning%20Web%20App/mushrooms.csv'
        df = pd.read_csv("C:\\Users\\M-TT\\Code\\ML Web App\\mushrooms.csv")
        label = LabelEncoder()
        for col in df.columns:
            df[col] = label.fit_transform(df[col])

        return df
    
    df = load_data()

    @st.cache_data
    def split(df):
        y = df['type']
        X = df.drop(['type'], axis=1)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test
    

    def plot_metrics(metrics):
        if 'Confusion Matrix' in metrics:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, ax=ax)
            st.pyplot(fig)

            

        if 'ROC Curve' in metrics:
            st.subheader("ROC Curve")
            fig, ax = plt.subplots()
            RocCurveDisplay.from_estimator(model, x_test, y_test, ax=ax)
            st.pyplot(fig)

        if 'Precision-Recall Curve' in metrics:
            st.subheader("Precision-Recall")
            fig, ax = plt.subplots()
            PrecisionRecallDisplay.from_estimator(model, x_test, y_test, ax=ax)
            st.pyplot(fig)

        
    x_train, x_test, y_train, y_test = split(df)
    class_names = ['edible', 'poisonous']
    classifier = st.sidebar.selectbox("Classifier", ["Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"])

    metrics = st.sidebar.multiselect("Plot Metric(s)", ['Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'])


    if classifier == "Support Vector Machine (SVM)":
        C = st.sidebar.number_input("C (Regularization Parameter)", 0.01, 10.0, step=0.01, key='C')
        kernel = st.sidebar.radio("Kernel", ["linear", "rbf"], key='kernel')
        gamma = st.sidebar.radio("Gamma", ["scale", "auto"], key='gamma')
       
        if st.sidebar.button("Enter", key='enter'):
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            yhat = model.predict(x_test)
            precision = precision_score(y_test, yhat)
            recall = recall_score(y_test, yhat)

            st.write("Accuracy: ", accuracy)
            st.write("Precision", precision)
            st.write("Recall: ", recall)
            plot_metrics(metrics)



    if classifier=="Logistic Regression":
        C = st.sidebar.number_input("C (Regularization Parameter)", 0.01, 10.0, step=0.01, key='C')
        max_iter = st.sidebar.slider("Maximum Iterations", 100, 500, key='max_iter')

        if st.sidebar.button("Enter", key='enter'):

            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            yhat = model.predict(x_test)
            precision = precision_score(y_test, yhat)
            recall = recall_score(y_test, yhat)

            st.write("Accuracy: ", accuracy)
            st.write("Precision", precision)
            st.write("Recall: ", recall)
            plot_metrics(metrics)


    if classifier=="Random Forest":
        n_estimator = st.sidebar.number_input("n_estimator", 100, 5000, step=10, key='n_estimator')
        max_depth = st.sidebar.number_input("max_depth", 1, 20, step=1, key='max_depth')
        bootstrap = st.sidebar.radio("bootstrap", ('True', 'False'), key='bootstrap')
        if st.sidebar.button("Enter", key='enter'):
            model = RandomForestClassifier(n_estimators=n_estimator, max_depth=max_depth)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            yhat = model.predict(x_test)
            precision = precision_score(y_test, yhat)
            recall = recall_score(y_test, yhat)

            st.write("Accuracy: ", accuracy)
            st.write("Precision", precision)
            st.write("Recall: ", recall)
            plot_metrics(metrics)

    if st.sidebar.checkbox("Display Training Data", False):
        st.subheader("Mushroom Data Set")
        st.write(df)




if __name__ == '__main__':
    main()



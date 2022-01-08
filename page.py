import streamlit as st
import pandas as pd
import time
from PIL import Image


st.title('Earthquake Prédicteur')

st.header('Overview')
st.success('The goal of the challenge is to capture the physical state of the laboratory fault and how close it is from failure from a snapshot of the seismic data it is emitting. You will have to build a model that predicts the time remaining before failure from a chunk of seismic data. This project predicts the likelihood of an earthquake to occur provided a test sample. Note this experiment can only make predictions based on the earthquake data provided by the Los Alamos National Lab.')
st.success('The data for this challenge comes from a classic laboratory earthquake experiment, that has been studied in depth as a tabletop analog of seismogenic faults for decades. A number of physical laws widely used by the geoscience community have been derived from this earthquake machine.')
st.info('- https://www.kaggle.com/c/LANL-Earthquake-Prediction/overview\n- https://doi.org/10.1002/2017GL074677\n- https://doi.org/10.1002/2017GL076708\n- https://rdcu.be/bdG8Y\n- https://rdcu.be/bdG9r')

st.header('How we built the model')

st.subheader('Data analysis')

st.success('- The training data is a single, continuous segment of experimental data.\n- The input is a chunk of 0.0375 seconds of seismic data (ordered in time), which is recorded at 4MHz, hence 150\'000 data points, and the output is time remaining until the following lab earthquake, in seconds.\n- The seismic data is recorded using a piezoceramic sensor, which outputs a voltage upon deformation by incoming seismic waves. The seismic data of the input is this recorded voltage, in integers.\n- The data is recorded in bins of 4096 samples. Withing those bins seismic data is recorded at 4MHz, but there is a 12 microseconds gap between each bin, an artifact of the recording device.')
st.markdown('Time range: `16.1 to 0 seconds`')

image = Image.open('images/top.png')
st.image(image, caption='Train data (top section)')
image = Image.open('images/bottom.png')
st.image(image, caption='Train data (bottom section)')
image = Image.open('images/middle.png')
st.image(image, caption='Train data (middle section)')


st.subheader('Feature engineering')

# with st.container():
#     col1, col2 = st.columns(2)
#     with col1:
#         st.markdown(f'### Time to next earthquake: `{5.61}s`')
#     with col2:
#         st.subheader('Magnitude:')
with st.container():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info('Mean\n- **Description**: The average signal data value amongst the block\n- **Reason for selection**: The expected signal value at a point in time is a standard statistic to measure when attempting to discover features about data trends in relation to the time till an earthquake.')
    with col2:
        st.info('STD\n- **Description**: The standard deviation of the signal data\n- **Reason for selection**: The spread of the data signals are also important in describing a signal\'s patterns and susceptibility to noise.')
    with col3:
        st.info('Min\n- **Description**: The minimum value\n- **Reason for selection**: Acoustic data contain many forms of noise, but often the noise has a gaussian (normal) distribution, so extreme values such as min and max can still be significant features for training.')
with st.container():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info('Max\n- **Description**: The maximum value \n- **Reason for selection**: Acoustic data contain many forms of noise, but often the noise has a gaussian (normal) distribution, so extreme values such as min and max can still be significant features for training.')
    with col2:
        st.info('Kurtosis\n- **Description**: The tailedness (behavior of the data distribution along the extreme ends)\n- **Reason for selection**: This statistic help indicate the spread of the data of the ends in particular, which is not covered specifically with just the standard deviation.')
    with col3:
        st.info('Skew\n- **Description**: The direction of asymmetry of the data’s distributions.\n- **Reason for selection**: Many of the previous statistics work extremely well when dealing with symmetrical and normally distributed data, but skew will help to describe scenarios where the signals are not truly symmetrical, which could be helpful in predicting.')
with st.container():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info('Quantile\n- **Description**: The data signal value where X% of all other signal values are smaller than it.\n- **Reason for selection**: ')
    with col2:
        st.info('IQR\n- **Description**: The range between the 25th quantile and the 75th quantile\n- **Reason for selection**: ')
    with col3:
        st.info('STD with no peaks\n- **Description**: The standard deviation of data signals points that within 2 standard deviations of the sample mean.\n- **Reason for selection**: ')
with st.container():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info('MFCC\n- **Description**: \n- **Reason for selection**: ')
    with col2:
        st.info('Power spectrum density\n- **Description**: \n- **Reason for selection**: ')
    with col3:
        st.info('Peak count\n- **Description**: The counted number of local signal peaks that are above desired threshold of 100\n- **Reason for selection**: ')
with st.container():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info('20th Quantile of Rolling standard deviation\n- **Description**: The 20th quantile of the distribution of standard deviations that are found through a rolling sampling of the data signals.\n- **Reason for selection**: ')
    with col2:
        st.info('80th Quantile of Rolling standard deviation\n- **Description**: The 80th quantile of the distribution of standard deviations that are found through a rolling sampling of the data signals.\n- **Reason for selection**: ')

st.subheader('Feature selection')
st.success('Catboost\'s `select_features`\n- Select the best features and drop harmful features from the dataset.\n- Removes complexity and increases interpretability')
st.info('- https://catboost.ai/en/docs/concepts/python-reference_catboost_select_features')

st.subheader('Model selection')
st.success('Catboost\n- Gradient boosting algorithm')

st.header('Demo application')

st.markdown(f'Training model accuracy: `{int(0.99*100)}%`')

st.warning('1. Uploade a csv file containing the acoustic data (typically around 150,000 data points)\n2. The time til\' the next predicted earthquake will be shown below in seconds.')


uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'##### Time til\' next earthquake: `{5.61}s`')
    with col2:
        st.markdown(f'##### Confidence: `{int(0.86*100)}%`')
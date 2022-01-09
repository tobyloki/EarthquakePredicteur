import streamlit as st
import time
from PIL import Image
from catboost import CatBoostRegressor
import pandas as pd
import numpy as np

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

with st.container():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info('##### Mean\n- **Description**: The average signal data value amongst the block\n- **Reason for selection**: The expected signal value at a point in time is a standard statistic to measure when attempting to discover features about data trends in relation to the time till an earthquake.')
    with col2:
        st.info('##### STD\n- **Description**: The standard deviation of the signal data\n- **Reason for selection**: The spread of the data signals are also important in describing a signal\'s patterns and susceptibility to noise.')
    with col3:
        st.info('##### Min\n- **Description**: The minimum value\n- **Reason for selection**: Acoustic data contain many forms of noise, but often the noise has a gaussian (normal) distribution, so extreme values such as min and max can still be significant features for training.')
with st.container():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info('##### Max\n- **Description**: The maximum value \n- **Reason for selection**: Acoustic data contain many forms of noise, but often the noise has a gaussian (normal) distribution, so extreme values such as min and max can still be significant features for training.')
    with col2:
        st.info('##### Peak count\n- **Description**: The counted number of local signal peaks that are above desired threshold of 100\n- **Reason for selection**: By counting the number of peaks, we can see if large number of peaks are indicators of trends in the data, rather than just noise and outliers.')
    with col3:
        st.info('##### Quantile\n- **Description**: The data signal value where X% of all other signal values are smaller than it.\n- **Reason for selection**: By only using specific quantile ranges, we are able to identify trends in specific ranges of the bell curve and eliminate outliers.')
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.info('##### Kurtosis\n- **Description**: The tailedness (behavior of the data distribution along the extreme ends)\n- **Reason for selection**: This statistic help indicate the spread of the data of the ends in particular, which is not covered specifically with just the standard deviation.')
    with col2:
        image = Image.open('images/kurtosis.png')
        st.image(image, caption='Kurtosis')
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.info('##### Skew\n- **Description**: The direction of asymmetry of the data’s distributions.\n- **Reason for selection**: Many of the previous statistics work extremely well when dealing with symmetrical and normally distributed data, but skew will help to describe scenarios where the signals are not truly symmetrical, which could be helpful in predicting.')
    with col2:
        image = Image.open('images/skew.png')
        st.image(image, caption='Skew')
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.info('##### IQR\n- **Description**: The range between the 25th quantile and the 75th quantile\n- **Reason for selection**: By only looking solely at the interquartile range, we can looking at the middle most data to identify trends in the bulk of the data.')
    with col2:
        st.info('##### STD with no peaks\n- **Description**: The standard deviation of data signals points that within 2 standard deviations of the sample mean.\n- **Reason for selection**: By eliminating the peaks before calculating standard deviation, we can eliminate outliers before calculating the standard deviation of the bell curve, allowing us to get a more accurate view of the trends of our data without excess noise.')
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.info('##### MFCC\n- **Description**: MFCC is used for processing sound, and is a representation of its short run power spectrum.\n- **Reason for selection**: Because all of the data is based on sound frequencies, it is critical to capture the short run trends of these freqencies in the power spectrum, so that we can identify if they are representative of a trend towards the time of failure.')
    with col2:
        image = Image.open('images/mfcc.png')
        st.image(image, caption='MFCC')
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.info('##### Power spectrum density\n- **Description**: The power spectrum density measures the power of sound singal\'s power against a certain frequency.\n- **Reason for selection**: The power spectrum density is critical for capture trends in the power of the sound signal in the data against certain frequency, thus allowing us to identify if these trends are indicative of a trend towards the time of failure.')
    with col2:
        image = Image.open('images/powerSpectrumDensity.png')
        st.image(image, caption='Power spectrum density')
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.info('##### 20th Quantile of Rolling standard deviation\n- **Description**: The 20th quantile of the distribution of standard deviations that are found through a rolling sampling of the data signals.\n- **Reason for selection**: By rolling through sample portions of the data, it helps to identify trends in the short run within a specific window and in the long run between separate windows. In addition, using the 20th quantile we are able to looking exclusively at the lowest values in the data to identify trends in this specific subset of data. In addition, by using standard deviation, we can identify if this data is getting more concentrated or more sparce within the quantile, allowing us to better identify if we are getting closer to the time of failure.')
    with col2:
        st.info('##### 80th Quantile of Rolling standard deviation\n- **Description**: The 80th quantile of the distribution of standard deviations that are found through a rolling sampling of the data signals.\n- **Reason for selection**: Similarly to the 20th Quantile of Rolling standard deviation, by rolling through sample portions of the data, it helps to identify trends in the short run within a specific window and in the long run between separate windows. In addition, using the 20th quantile we are able to looking exclusively at the lowest values in the data to identify trends in this specific subset of data. In addition, by using standard deviation, we can identify if this data is getting more concentrated or more sparce within the quantile, allowing us to better identify if we are getting closer to the time of failure.')
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        image = Image.open('images/rollingWindow.png')
        st.image(image, caption='Rolling window')
    with col2:
        image = Image.open('images/rollingWindowStd.png')
        st.image(image, caption='Rolling standard deviation')

st.subheader('Feature selection')
st.success('##### Catboost\'s `select_features`\n- Select the best features and drop harmful features from the dataset.\n- Removes complexity and increases interpretability\n- Catboost utilizes a function called select_features that allows it to take in a collection of input training data and possible features, and then returns back which features in the most reliable and should be selected for use in this model after a series of many iterations. In addition, it also makes an effort to avoid overfitting and monitors changes in metrics.')
st.info('- https://catboost.ai/en/docs/concepts/python-reference_catboost_select_features')

st.subheader('Model selection')
st.success('##### What is Gradient Boosting?\n- Method for both regression and classification problems\n- Produces prediction model as a collection of many weak prediction models (typically decision trees)\n- When new model added, all models reweighted so stronger models are weighted higher and weaker models are weighted lower\n- Thus iteratively improves upon many weak prediction models to create one strong aggregate model that is better than the sum of its parts.\n\n##### What is CatBoost?\n- Version of Gradient Boosting, tends to work very well compared to similar models\n- Automatically converts categorical values into numbers\n- Then able to combine categorical and numerical features\n- Combines many predictors together, giving weights to each, for a single aggregate model')

st.subheader('Model metrics')

metrics = pd.read_csv('EarthquakeModelMetrics.csv').iloc[0]
score = metrics[0] * 100
mae = metrics[1]

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Accuracy", value=f"{score:.2f}%")
    with col2:
        st.metric(label="Loss (MAE)", value=f"{mae:.4f}")

st.header('Demo application')

model = CatBoostRegressor()
model.load_model('EarthquakeModel.cbm')

def gen_features(X):
    strain = []
    strain.append(X.mean())
    strain.append(X.std())
    strain.append(X.min())
    strain.append(X.max())
    strain.append(X.kurtosis())
    strain.append(X.skew())
    strain.append(np.quantile(X,0.95))
    strain.append(np.quantile(X,0.90))
    strain.append((X.loc[abs(X - X.mean()) < 20]).kurtosis()) # truncated kurtosis 
    strain.append(np.quantile(X,0.75) - np.quantile(X,0.25)) #iqr
    ### std_nopeak: https://stackoverflow.com/questions/51006163/pandas-how-to-detect-the-peak-points-outliers-in-a-dataframe
    df = X.copy(deep = True) #temp df
    from scipy import stats
    df_Z = df[(np.abs(stats.zscore(df)) < 2.0)] # Use z-score of 2 to remove peaks
    ix_keep = df_Z.index
    df_keep = df.loc[ix_keep] # Subset the raw dataframe with the indexes you'd like to keep
    strain.append(df_keep.std())
    ### mfcc - https://www.kaggle.com/ilu000/1-private-lb-kernel-lanl-lgbm/
    # import librosa
    # mfcc = librosa.feature.mfcc(X.values)
    # strain.append(mfcc.mean(axis=1))
    ### power spectrum
    from scipy.signal import find_peaks, periodogram
    fs = 1.0
    f, Pxx_spec = periodogram(X, fs, 'flattop')
    strain.append(sum(Pxx_spec)/len(Pxx_spec)) #mean!
    # Peaks and rolling std
    strain.append(len(find_peaks(X, height=100)[0])) # peak count
    strain.append(np.quantile(X.rolling(50).std().dropna(), 0.2)) # rolling stdev
    return pd.Series(strain)


st.warning('1. Uploade a csv file containing the acoustic data (typically around 150,000 data points)\n2. The time til\' the next predicted earthquake will be shown below in seconds.')

preds = 0

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    test = pd.read_csv(uploaded_file)
    st.write(test)

    X_test = pd.DataFrame()
    y_test = pd.Series(dtype=np.float64)
    ch = gen_features(test['acoustic_data'])
    X_test = X_test.append(ch, ignore_index=True)

    # Making prediction 
    preds = model.predict(X_test)[0]

st.markdown(f'##### Time til\' next earthquake: `{preds:.2f}s`')

import streamlit as st
import pickle
import sklearn
import pandas as pd
import numpy as np
from PIL import Image
import json

model_name = 'model_KNNr.pkl'
model = pickle.load(open(model_name, 'rb'))

st.title('TJM - Data Scientist Freelance')
# st.sidebar.header('Compétences')
image = Image.open('550-TJM.jpg')
st.image(image, '')

# FUNCTION

def user_report():
  st.sidebar.header('Lieu de résidence')
  val_auvergne, val_bretagne, val_hfrance, val_normandie, val_naquitaine, val_occitanie, val_paca, val_idf = 1, 0, 0, 0, 0, 0, 0, 0
  location = st.sidebar.selectbox('Où résidez-vous ou région proche ?',('Auvergne-Rhône-Alpes', 'Bretagne', 'Hauts-de-France', 'Normandie', 'Nouvelle-Aquitaine', 'Occitanie', 'Provence-Alpes-Côte dAzur', 'Île-de-France'))
  if location == 'Auvergne-Rhône-Alpes':
    val_auvergne, val_bretagne, val_hfrance, val_normandie, val_naquitaine, val_occitanie, val_paca, val_idf = 1, 0, 0, 0, 0, 0, 0, 0
  elif location == 'Bretagne':
    val_auvergne, val_bretagne, val_hfrance, val_normandie, val_naquitaine, val_occitanie, val_paca, val_idf = 0, 1, 0, 0, 0, 0, 0, 0
  elif location == 'Hauts-de-France':
    val_auvergne, val_bretagne, val_hfrance, val_normandie, val_naquitaine, val_occitanie, val_paca, val_idf = 0, 0, 1, 0, 0, 0, 0, 0
  elif location == 'Normandie':
    val_auvergne, val_bretagne, val_hfrance, val_normandie, val_naquitaine, val_occitanie, val_paca, val_idf = 0, 0, 0, 1, 0, 0, 0, 0
  elif location == 'Nouvelle-Aquitaine':
    val_auvergne, val_bretagne, val_hfrance, val_normandie, val_naquitaine, val_occitanie, val_paca, val_idf = 0, 0, 0, 0, 1, 0, 0, 0
  elif location == 'Occitanie':
    val_auvergne, val_bretagne, val_hfrance, val_normandie, val_naquitaine, val_occitanie, val_paca, val_idf = 0, 0, 0, 0, 0, 1, 0, 0
  elif location == 'Provence-Alpes-Côte dAzur':
    val_auvergne, val_bretagne, val_hfrance, val_normandie, val_naquitaine, val_occitanie, val_paca, val_idf = 0, 0, 0, 0, 0, 0, 1, 0
  elif location == 'Île-de-France':
    val_auvergne, val_bretagne, val_hfrance, val_normandie, val_naquitaine, val_occitanie, val_paca, val_idf = 0, 0, 0, 0, 0, 0, 0, 1

  st.sidebar.header('Expérience')
  val_advanced, val_beginner, val_intermediate, val_junior, val_senior = 1, 0, 0, 0, 0
  experience = st.sidebar.selectbox("Nombre d'années d'expérience ?",('Advanced : 7+ ans', 'Beginner : 1+ ans', 'Intermediate : 4+ ans', 'Junior : 2+ ans', 'Senior'))
  if experience == 'Advanced : 7+ ans':
    val_advanced, val_beginner, val_intermediate, val_junior, val_senior = 1, 0, 0, 0, 0
  elif experience == 'Beginner : 1+ ans':
    val_advanced, val_beginner, val_intermediate, val_junior, val_senior = 0, 1, 0, 0, 0
  elif experience == 'Intermediate : 4+ ans':
    val_advanced, val_beginner, val_intermediate, val_junior, val_senior = 0, 0, 1, 0, 0
  elif experience == 'Junior : 2+ ans':
    val_advanced, val_beginner, val_intermediate, val_junior, val_senior = 0, 0, 0, 1, 0
  elif experience == 'Senior : 10+ ans' :
    val_advanced, val_beginner, val_intermediate, val_junior, val_senior = 0, 0, 0, 0, 1

  st.sidebar.header('Compétences')
  val_eval = st.sidebar.radio('Entretien technique ?', ['no', 'yes'])
  val_python = st.sidebar.radio('Python', ['no', 'yes'])
  val_ml = st.sidebar.radio('Machine Learning', ['no', 'yes'])
  val_ds = st.sidebar.radio('Data SCience', ['no', 'yes'])
  val_sql = st.sidebar.radio('SQL', ['no', 'yes'])
  val_dl = st.sidebar.radio('Deep Learning', ['no', 'yes'])
  val_da = st.sidebar.radio('Data Analysis', ['no', 'yes'])
  val_bd = st.sidebar.radio('Big Data', ['no', 'yes'])
  val_r = st.sidebar.radio('R', ['no', 'yes'])
  val_sl = st.sidebar.radio('Scikit Learn', ['no', 'yes'])
  val_pd = st.sidebar.radio('Pandas', ['no', 'yes'])
  val_spark = st.sidebar.radio('Spark', ['no', 'yes'])
  val_java = st.sidebar.radio('Java', ['no', 'yes'])
  val_cplus = st.sidebar.radio('C++', ['no', 'yes'])
  val_tf = st.sidebar.radio('TensorFlow', ['no', 'yes'])
  val_hadoop = st.sidebar.radio('Hadoop', ['no', 'yes'])
  val_keras = st.sidebar.radio('Keras', ['no', 'yes'])
  val_dv = st.sidebar.radio('Data Visualization', ['no', 'yes'])
  val_matlab = st.sidebar.radio('Matlab', ['no', 'yes'])
  val_js = st.sidebar.radio('JavaScript', ['no', 'yes'])
  val_np = st.sidebar.radio('Numpy', ['no', 'yes'])
  val_linux = st.sidebar.radio('Linux', ['no', 'yes'])
  val_m = st.sidebar.radio('Management', ['no', 'yes'])
  val_nlp = st.sidebar.radio('NLP', ['no', 'yes'])
  val_sas = st.sidebar.radio('SAS', ['no', 'yes'])
  val_bi = st.sidebar.radio('Business Intelligence', ['no', 'yes'])
  val_nosql= st.sidebar.radio('NoSQL', ['no', 'yes'])
  val_ai = st.sidebar.radio('Artificial Intelligence', ['no', 'yes'])
  val_scala = st.sidebar.radio('Scala', ['no', 'yes'])
  val_vba = st.sidebar.radio('VBA', ['no', 'yes'])
  val_flask = st.sidebar.radio('Flask', ['no', 'yes'])
  val_pb = st.sidebar.radio('Power BI', ['no', 'yes'])
  val_aws = st.sidebar.radio('AWS', ['no', 'yes'])
  val_c = st.sidebar.radio('Consulting', ['no', 'yes'])
  val_algo = st.sidebar.radio('Algorithms', ['no', 'yes'])
  val_mat = st.sidebar.radio('Matplotlib', ['no', 'yes'])
  
  user_report_data = {
      'evaluated': 1 if val_eval == 'yes' else 0,
      'python' : 1 if val_python == 'yes' else 0,
      'machine learning' : 1 if val_ml == 'yes' else 0,
      'data science' : 1 if val_ds == 'yes' else 0,
      'sql' : 1 if val_sql == 'yes' else 0,
      'deep learning' : 1 if val_dl == 'yes' else 0,
      'data analysis' : 1 if val_da == 'yes' else 0,
      'big data' : 1 if val_bd == 'yes' else 0,
      'r' : 1 if val_r == 'yes' else 0,
      'scikit learn' : 1 if val_sl == 'yes' else 0,
      'pandas' : 1 if val_pd == 'yes' else 0,
      'spark' : 1 if val_spark == 'yes' else 0,
      'java' : 1 if val_java == 'yes' else 0,
      'c++' : 1 if val_cplus == 'yes' else 0,
      'tensorflow' : 1 if val_tf == 'yes' else 0,
      'hadoop' : 1 if val_hadoop == 'yes' else 0,
      'keras' : 1 if val_keras == 'yes' else 0,
      'data visualization' : 1 if val_dv == 'yes' else 0,
      'matlab' : 1 if val_matlab == 'yes' else 0,
      'javascript' : 1 if val_js == 'yes' else 0,
      'numpy' : 1 if val_np == 'yes' else 0,
      'linux' : 1 if val_linux == 'yes' else 0,
      'management' : 1 if val_m == 'yes' else 0,
      'nlp' : 1 if val_nlp == 'yes' else 0,
      'sas' : 1 if val_sas == 'yes' else 0,
      'business intelligence' : 1 if val_bi == 'yes' else 0,
      'nosql' : 1 if val_nosql == 'yes' else 0,
      'artificial intelligence' : 1 if val_ai == 'yes' else 0,
      'scala' : 1 if val_scala == 'yes' else 0,
      'vba' : 1 if val_vba == 'yes' else 0,
      'flask' : 1 if val_flask == 'yes' else 0,
      'power bi' : 1 if val_pb == 'yes' else 0,
      'aws' : 1 if val_aws == 'yes' else 0,
      'consulting' : 1 if val_c == 'yes' else 0,
      'algorithms' : 1 if val_algo == 'yes' else 0,
      'matplotlib' : 1 if val_mat == 'yes' else 0,
      'region_Auvergne-Rhône-Alpes': val_auvergne,
      'region_Bretagne': val_bretagne,
      'region_Hauts-de-France': val_hfrance,
      'region_Normandie': val_normandie,
      'region_Nouvelle-Aquitaine': val_naquitaine, 
      'region_Occitanie': val_occitanie,
      'region_Provence-Alpes-Côte dAzur': val_paca, 
      'region_Île-de-France': val_idf,
      'experience_advanced': val_advanced, 
      'experience_beginner' : val_beginner, 
      'experience_intermediate' : val_intermediate, 
      'experience_junior' : val_junior, 
      'experience_senior': val_senior
  }
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data

user_data = user_report()
st.header('Vos données au format Dataframe : ')
st.write(user_data)

# View as JSON
user_data_json = user_data.to_json(orient="index")

with st.beta_expander("🔍: Vos données au format JSON "):
	st.json(user_data_json)

# user_data_json = user_data.to_json(orient="index")
# st.json(user_data_json)

# with st.beta_expander("📩: Download"):
	# make_downloadable_df_format(df,dataformat)

salary = model.predict(user_data)
st.subheader('TJM estimé :')
st.subheader(str(np.round(salary[0], 2))+' €')
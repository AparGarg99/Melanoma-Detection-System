
############################################ IMPORT LIBRARIES ################################################
import streamlit as st
import PIL.Image
from urllib.request import urlopen
import numpy as np
import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta
import tensorflow as tf
import os
import sys
import cv2
import matplotlib.pyplot as plt
from fpdf import FPDF
import PyPDF2
import smtplib
import imghdr
from email.message import EmailMessage
import requests
from tensorflow.keras.models import Model, load_model
from scipy import ndimage
import math
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import resize
from final_inference_code import *
from PIL import Image

today = date.today()
##############################################################################################################
################################################ SIDEBAR #####################################################
##############################################################################################################

############################### Contact Us link ################################
st.sidebar.markdown(
    """<a style='display: block; text-align: right;' href="https://github.com/AparGarg99"><b>Contact Us</b></a>
    """,
    unsafe_allow_html=True,
)

######################## Take skin lesion input image ###########################
st.sidebar.title('Skin Lesion Image')
st.sidebar.warning("""Please upload image""")
uploaded_file = st.sidebar.file_uploader("Upload image", type=['png', 'jpg'])

# black: https://i.pinimg.com/originals/8e/ea/46/8eea4621a56088e49847c1f03c2aa337.jpg
# white: https://ak.picdn.net/shutterstock/videos/339694/thumb/1.jpg?ip=x480

sample_url = 'http://www.pngmagic.com/product_images/solid-dark-grey-background.jpg'

try:

    if uploaded_file is None:
        img = PIL.Image.open(urlopen(str(sample_url)))
        
    else:
        img = PIL.Image.open(uploaded_file)
          
except:
    st.sidebar.error("Invalid Image...try again!!")
    img = PIL.Image.open(urlopen(sample_url))


########################## Details for Report Generation ########################

######### PATIENT INFORMATION #########
st.sidebar.title('Patient Information')
p_name = st.sidebar.text_input('Full Name','',key='patient_name')
p_sex = st.sidebar.radio('Sex', ['M','F'])
p_dob = st.sidebar.date_input('DOB', value=None, 
                                    min_value=today-relativedelta(years=100) , 
                                    max_value=today)
p_email = st.sidebar.text_input('Email ID','',key='patient_email')

######## PHYSICIAN INFORMATION ########
st.sidebar.title('Physician Information')
phys_name = st.sidebar.text_input('Full Name','',key='physician_name')
phys_name = 'Dr. '+phys_name
phys_email = st.sidebar.text_input('Email ID','',key='physician_email')

####### SPECIMEN INFORMATION ##########
#st.sidebar.title('Specimen Information')
collected = str(today) #st.sidebar.date_input('Specimen Collected')
received = collected
reported = collected

##############################################################################################################
########################################## BACKEND PREDICTION ################################################
##############################################################################################################

def main_fuc(image):
    model_path = '.'
    model_segmentation_name = 'unet_large_jaccard_with3Augmentations.hdf5'
    model_arbitrator_name = 'Final_Arbitrator_6lyr_binaryCross_Weight5_YesAug_0.5Dropout.hdf5'
    UnetModel,model_arbitrator = load_all_models(model_path,model_segmentation_name,model_arbitrator_name)
    oErrorFlag,asym_img,border_img,center_img,whole_img = perform_all_preprocessing(image,UnetModel)
    if oErrorFlag == 0:
        output = model_arbitrator.predict([whole_img,asym_img, border_img, center_img,whole_img])
        melanomaProbability = 1 - output
    else:
        melanomaProbability = [[-1]]
    
    return melanomaProbability 

try:
  img2 = img.resize((600,450), Image.ANTIALIAS)
  prob = main_fuc(np.array(img2))[0][0]
  #st.error(prob)
except Exception as e:
  st.error(e)
  prob = -1

if(prob>0.5):
  pred = 'melanoma'
elif(0<=prob<=0.5):
  pred='non-melanoma'
else:
  pred=''


##############################################################################################################
########################################## BACKEND REPORT GENERATION #########################################
##############################################################################################################

######################### SETUP ###################################
# save FPDF() class into a variable pdf
pdf = FPDF()

# Add a page
pdf.add_page()

# Import Template
pdf.image('OurGroup_Template.jpg', x = 0, y = 0, w = 210, h = 297)
 
######################### Title text ###################################
pdf.set_font('Arial', 'B', 30)
pdf.set_text_color(255, 255, 255)
pdf.text(x=30, y=20, txt='Melanoma Detection System')

###################### PATIENT INFORMATION ###############################
# Name text
pdf.set_font('Arial',size=12)
pdf.set_text_color(0, 0, 0)
pdf.text(x=20, y=47, txt=p_name)

# Gender text
pdf.text(x=16, y=53.5, txt=p_sex)

# DoB text
pdf.text(x=17, y=60, txt=str(p_dob))

###################### PHYSICIAN INFORMATION ###############################
# Name text
pdf.text(x=92, y=47, txt=phys_name)

###################### SPECIMEN INFORMATION ###############################
pdf.text(x=172, y=53.5, txt=collected)

pdf.text(x=172, y=60, txt=collected)

pdf.text(x=172, y=66.5, txt=collected)

############################# DIAGNOSIS ####################################
# Lesion Image Text
pdf.text(x = 70, y = 100, txt='Skin Lesion')

# Lesion Image
img_name="lesion_image.jpg"
img.save(img_name)
pdf.image(img_name, x = 70, y = 105, w = 64, h = 64)

# Probability plot text
pdf.text(x = 70, y = 180, txt='Prediction')

# Probability plot
fig, ax = plt.subplots( nrows=1, ncols=1 ) 
ax.set_ylabel('Probability')
ax.set_xlabel('Class')
ax.bar(['melanoma','non-melanoma'],[prob,1-prob])
fig.savefig('prob_plot.png')   
plt.close(fig)
pdf.image('prob_plot.png', x = 55, y = 185, w = 90, h = 80)

############################ SAVE REPORT ##################################
pdf.output("Report.pdf")


##############################################################################################################
######################################## BACKEND EMAILING FEATURE #############################################
##############################################################################################################

############################ VERIFY EMAIL ID ##################################
def verify_email(email_id):
  try:
    # https://hunter.io/api-keys
    API_KEY = ""
    id = email_id
    response = requests.get("https://api.hunter.io/v2/email-verifier?email={}&api_key={}".format(id,API_KEY))
    resp = response.json()['data']

    if(resp['result']!='undeliverable' and resp['regexp']==True and resp['gibberish']==False):
      check = 'Valid'
    else:
      check = 'Invalid'

  except:
    check = 'Invalid'

  return check

def verify_error_msgs(check1,check2):
  if(check1=='Invalid' and check2=='Valid'):
    email_list = [phys_email]

  elif(check1=='Valid' and check2=='Invalid'):
    email_list = [p_email]

  elif(check1=='Invalid' and check2=='Invalid'):
    email_list = []

  else:
    email_list=[p_email,phys_email]

  return email_list

############################ SEND MAIL ##################################
def send_mail():
  try:
    check1 = verify_email(p_email)
    check2 = verify_email(phys_email)
    # st.write(check1)
    # st.write(check1)
    email_list = verify_error_msgs(check1,check2)

    if(email_list!=[]):
      EMAIL_ADDRESS = ""
      EMAIL_PASSWORD = ""

      msg = EmailMessage()
      msg['Subject'] = 'Skin Cancer Report'
      msg['From'] = EMAIL_ADDRESS
      msg['To'] = ", ".join(email_list)

      msg.set_content('Hi {},\n\n Please find the skin cancer report attached above.\n\n Thanks,\n {}'.format(p_name,phys_name))

      file='Report.pdf'
      with open(file,'rb') as f:
        file_data = f.read()
        file_name = f.name

      msg.add_attachment(file_data,maintype="application",subtype="octet-stream",filename=file_name)

      with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
          smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
          smtp.send_message(msg)

  except:
    pass  


##############################################################################################################
################################################ MAIN PAGE ###################################################
##############################################################################################################

###################### Display title of the project #######################
st.markdown("<h1 style='text-align: center; color: #DBDE0A; font-size:400%; font-family:Brush Script MT, cursive;'>Melanoma Detection System</h1>", unsafe_allow_html=True)
st.write('')

######################## Display app description ##########################
expander_bar = st.expander("About App")
expander_bar.markdown('''
	* This project uses a Deep CNN to classify images of skin lesions as melanoma/non-melanoma.
	* Read More : https://github.com/AparGarg99/Melanoma-Detection-System
		''')
st.write('')

######################## Display user input image #########################
st.markdown("<p style= 'color: #DBDE0A; font-size:190%;'>Image you've selected</p>", unsafe_allow_html=True)

img = img.resize((224,224))
st.image(img)
st.write('')

#################### Display model predicted food class ###################
st.markdown("<p style= 'color: #DBDE0A; font-size:190%;'>Prediction</p>", unsafe_allow_html=True)

if uploaded_file is not None:
  st.write(pred.replace('_', ' ').title())

else:
  st.write('')

st.write('')

########################## Download and Email Report ###############################
if uploaded_file is not None and pred!='':
  st.download_button(label="Download & Email Report",data = open("Report.pdf", 'rb'), file_name="Downloaded_Report.pdf",on_click = send_mail)
  

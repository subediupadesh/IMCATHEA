# IMCATHEA
IMCATHEA is a python based Graphical User Interface application capable of detecting the IM phase in a High Entropy Alloy or Multi Principal Element Alloy system using a Neural Network Algorithm.

## For Linux User:
Python with following libraries are needed to run the GUI application: pymatgen (v2022.0.16), matminer (v0.7.4), tensorflow (v2.2.0), pandas (v1.3.4), numpy (v1.21.2), tkinter (v8.6), sklearn (v1.0.1). streamlit(v1.25.0)

"IMCATHEA_GUI.py" also needs 4 extra supporitng files (icon, standardizations, and model) to operate, which will be made available from the author upon reasonable request. 


## For Web APP:
Visit the website : https://imcathea.streamlit.app

And follow the steps below to run the app

## Procedure for using "IMCATHEA" GUI Application:

Step 0. Open / Run "IMCATHEA_GUI.py" to open the GUI application

Step 1. User need to select the No. of Elements/Components in the HEA where IM phase is to be detected, from the dropdown menu at the top left corner.
		    After selection of element size/number (from 2 to 10),
		    
Step 2. User can select each element of HEA, one at a time from the drop down menu generated just below "No. of Component" tab

Step 3. After selection of each element a blank space is provided just at the right side of selected element tab where user need to enter the corresponding composition/elemental           fraction of the element.

Step 4. Repeat Step 2-3 until the last element of the HEA and it's composition/elemental fraction is entered.

Step 5. Press "Detect IMC" tab to get the prediction if IM is present in the provided HEA or not along with the physical properties of the entered HEA displayed in the right side of the GUI application.

Step 6. If user wants to detect the presence or absence of IM phase for another HEA, user can click on "Restart" tab at the right top side of GUI application to restart the application instantly.


If the user makes errors while selecting any options from dropdown menu as suggested in Step 1,2,3,4 user can click on "Restart" tab at the right top side of GUI application to restart the application instantly and start the process for prediction again from Step 1.



## Video (YouTube) Tutorial for using IMCATHEA:
The stepwise process to predict the phase for an example case of "CrNbTiZr" MPEA is shown in the YouTube video tutorial at: https://youtu.be/G3aHE1Wsmgk

## How to cite IMCATHEA libraries:
If you use IMCATHEA in your research, please cite:

Subedi, U.; Coutinho, Y.A.; Malla, P.B.; Gyanwali, K.; Kunwar, A. Automatic Featurization Aided Data-Driven Method for Estimating the Presence of Intermetallic Phase in Multi-Principal Element Alloys. Metals 2022, 12, 964. https://doi.org/10.3390/met12060964

## For the Full Open Access Paper: 
https://doi.org/10.3390/met12060964

### Code by Upadesh Subedi
## If you use the result from this code or any part of this work, Please cite the paper:
    ## Subedi, U.; Coutinho, Y.A.; Malla, P.B.; Gyanwali, K.; Kunwar, A. Automatic Featurization Aided Data-Driven Method for Estimating the Presence of Intermetallic Phase in Multi-Principal Element Alloys. Metals 2022, 12, 964. 
    ## https://doi.org/10.3390/met12060964

import sys, os
import streamlit as st
import numpy as np
import pandas as pd
from keras.layers import LeakyReLU, ReLU, Softmax
from pickle import load
from keras.models import load_model
from sklearn.preprocessing import StandardScaler, RobustScaler

from pymatgen.core.composition import Composition, Element
from matminer.featurizers.composition.alloy import Miedema, WenAlloys, YangSolidSolution
from matminer.featurizers.composition import ElementFraction
from matminer.featurizers.conversions import StrToComposition
from matminer.utils.data import MixingEnthalpy, DemlData
from matminer.utils import data_files #for Miedema.csv present inside package

ef = ElementFraction()
stc = StrToComposition()


def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


def restart():
     st.session_state.clear()

st.set_page_config(layout="wide")
st.title('IMCATHEA')

elements = ['Li', 'Be', 'B', 'Na', 'Mg', 'Al', 'Si', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ge', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Ru', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'La', 'Ce', 'Nd', 'Gd', 'Yb', 'Hf', 'Ta', 'W', 'Au', 'Bi']
thermo_values = ['Atomic size diff (δ)', 'ΔHmix', 'ΔSmix', 'Omega (Ω)', 'Δχ', 'VEC']

element_symbols = ['Click to Select','Ag', 'Al', 'Au', 'B', 'Be', 'Bi', 'Ca', 'Cd', 'Ce', 'Co', 'Cr', 'Cu', 'Fe', 'Gd', 'Ge', 'Hf', 'In', 'La', 'Li', 'Mg', 'Mn', 'Mo', 'Na', 'Nb', 'Nd', 'Ni', 'Pd', 'Ru', 'Sb', 'Sc', 'Si', 'Sn', 'Sr', 'Ta', 'Ti', 'V', 'W', 'Y', 'Yb', 'Zn', 'Zr']

main_col1, main_col2 = st.columns(2)

s_col11, s_col12, s_col13 = main_col1.columns(3)
s_col21, s_col22 = main_col2.columns(2)

No_Comp_txt = '<p style="color:White; font-size: 30px;">Select No of <br> Components:</p>'
s_col11.markdown(No_Comp_txt, unsafe_allow_html=True)


num_rows = s_col12.selectbox('', options=list(range(2, 11)), index=0)

s_col11, s_col12, s_col13 = main_col1.columns(3) # levelling the row at same position 
selected_elements = []
selected_compositions = []

for i in range(num_rows):
    available_elements = element_symbols.copy()
    for element in selected_elements:
            if (element in available_elements and element != available_elements[0]):  # available_elements[0] => 'Click to Select'
                available_elements.remove(element)

    selected_element = s_col11.selectbox(f'Element {i+1}:', available_elements, key=f'element:{i}')
    element_composition = s_col12.number_input(f'Enter Composition' if selected_element==available_elements[0] else f'Enter Composition of {selected_element}', value=1.0, min_value=1e-29, key=f'value:{i}')
    selected_elements.append(selected_element)
    selected_compositions.append(element_composition)

heading_txt_prop = '<p style="color:White; font-size: 35px;">Calculated Properties:</p>'
heading_values = '<p style="color:White; font-size: 35px;">Values:</p>'
s_col21.markdown(heading_txt_prop, unsafe_allow_html=True)
s_col22.markdown(heading_values, unsafe_allow_html=True)

prop_2 = s_col21.markdown(f"Atomic Size difference <span style='margin-right: 78px;'>&nbsp;</span> (&delta;)", unsafe_allow_html=True)
prop_2 = s_col21.markdown(f"Enthalpy of Mixing <span style='margin-right: 85px;'>&nbsp;</span> (&Delta; Hmix)", unsafe_allow_html=True)
prop_2 = s_col21.markdown(f"Entropy of Mixing <span style='margin-right: 95px;'>&nbsp;</span> (&Delta; Smix)", unsafe_allow_html=True)
prop_2 = s_col21.markdown(f"Temperature of Mixing <span style='margin-right: 70px;'>&nbsp;</span> (&Delta; Tm)", unsafe_allow_html=True)
prop_2 = s_col21.markdown(f"Omega Parameter <span style='margin-right: 108px;'>&nbsp;</span> (&Omega; = Tm(&Delta; Smix/|&Delta; Hmix|))", unsafe_allow_html=True)
prop_2 = s_col21.markdown(f"Electronegativity <span style='margin-right: 113px;'>&nbsp;</span> (&Delta;&Chi;)", unsafe_allow_html=True)
prop_2 = s_col21.markdown(f"Valence Electron Concentration <span style='margin-right: 15px;'>&nbsp;</span> (VEC)", unsafe_allow_html=True)

st.divider()


#############
### MPEA name  
full = []
full_HEA = []
for i in range(num_rows):
    if selected_elements[i] !='Click to Select':
        if (selected_compositions[i]=='1' or selected_compositions[i]==1.00):
             full.append(selected_elements[i])
             full_HEA.append(selected_elements[i])
        else:
             full.append(selected_elements[i]+'$_{%s}$'%(selected_compositions[i]))
             full_HEA.append(selected_elements[i]+f'{selected_compositions[i]}')
full_hea = ' '.join(full)
full_MPEA = ' '.join(full_HEA)
#############


#############
### Checking I/P composition correctness
for i, value_comp in enumerate(selected_compositions):
    if value_comp =="":
        st.write(" \n ")
    else:
        try:
            value_comp = float(value_comp)
            if float(value_comp) <= 0:
                st.title("COMPOSITION ERROR: Every entered composition should be greater than 0")
                break
        except:
            st.title(f"ERROR: Entered composition for Element {i+1} is INVALID")
            break
#############


elem_prop_data = pd.read_csv(os.path.dirname(data_files.__file__) +"/Miedema.csv", na_filter = False) #for Miedema.csv present inside package
VEC_elements = elem_prop_data.set_index('element')['valence_electrons'].to_dict()


#############
### Prediction 
predict_button = st.button("DETECT IMC")
# s_col211 = main_col2([2,2])
if predict_button:  
        HEA = [str(i) for i in selected_elements]
        HEA_ = [float(i) for i in selected_compositions]
        composition = dict.fromkeys(elements, 0)

        mole_fraction = []
        X_i = []
        r_i = []
        Tm_i = []
        VEC_i =[]
        R = 8.314
        for i in HEA:
            mole_fraction.append(Composition(full_MPEA).get_atomic_fraction(i))
            X_i.append(Element(i).X)
            if Element(i).atomic_radius_calculated == None:
                 r_i.append(Element(i).atomic_radius)
            else:
                 r_i.append(Element(i).atomic_radius_calculated)
            Tm_i.append(Element(i).melting_point)
            try: VEC_i.append(DemlData().get_elemental_property(Element(i), "valence"))
            except KeyError:
                if i in VEC_elements: VEC_i.append(float(VEC_elements.get(i)))
            
        # st.write(VEC_i)

        for i in range(len(HEA)):
                    if HEA[i] in composition:
                        composition.update({HEA[i]:mole_fraction[i]})
        composition_list = list(composition.values())
        attributes = list(composition.values())  

        # Atomic Size Difference Calculation
        r_bar = sum(np.multiply(mole_fraction, r_i))
        term = (1-np.divide(r_i, r_bar))**2
        atomic_size_difference = sum(np.multiply(mole_fraction, term))**0.5
        attributes.append(atomic_size_difference)


        # Enthalpy of Mixing Calculation
        AB = []
        C_i_C_j = []
        del_Hab = []
        for i in range(len(HEA)):
            for j in range(i, len(HEA)-1):
                AB.append(HEA[i] + HEA[j+1])
                C_i_C_j.append(mole_fraction[i]*mole_fraction[j+1])
                del_Hab.append(round(Miedema().deltaH_chem([HEA[i], HEA[j+1]], [0.5, 0.5], 'ss'),3))
        #         del_Hab.append(MixingEnthalpy().get_mixing_enthalpy(Element(HEA[i]), Element(HEA[j+1]))) # Matminer MixingOfEnthalpy
        omega = np.multiply(del_Hab, 4)
        del_Hmix = sum(np.multiply(omega, C_i_C_j))

        attributes.append(del_Hmix)

        # Entropy of Mixing Calculation
        del_Smix = -WenAlloys().compute_configuration_entropy(mole_fraction)*1000
        # del_Smix = -R*sum(np.multiply(mole_fraction, np.log(mole_fraction)))

        attributes.append(del_Smix)


        # Average Melting Temperature Calculation        
        Tm = sum(np.multiply(mole_fraction, Tm_i))

        # Omega parameter Calculation
        Omega = (Tm*del_Smix)/abs(del_Hmix*1000)         # Converting Kilo Joules of del_Hmix to joules 

        attributes.append(Omega)

        # Electronegativity Calculation                
        X_bar = sum(np.multiply(mole_fraction, X_i))
        del_Chi = (sum(np.multiply(mole_fraction, (np.subtract(X_i, X_bar))**2)))**0.5

        attributes.append(del_Chi)

        # Valence Electron Concentration Calculation
        VEC = sum(np.multiply(mole_fraction, VEC_i))

        attributes.append(VEC)

        column_names = elements + thermo_values 

        attributes = pd.DataFrame([attributes], columns= column_names)

        model_path = resource_path('model.h5')
        robust_std_path = resource_path('robust_standardization.pkl')
        scaler_path = resource_path('scaler_standardization.pkl')

        saved_model = load_model(model_path, custom_objects={'LeakyReLU': LeakyReLU(), 'ReLU': ReLU(), 'Softmax': Softmax()}, compile =False)

        input_attributes = 'Y' if int(saved_model.get_config().get('layers')[0].get('config').get('batch_input_shape')[1]) ==47 else 'N'
       
        if input_attributes == 'N':
            robust_1 = load(open(robust_std_path,'rb'))
            properties = attributes[thermo_values]
            pro = robust_1.transform(properties)
            attributes = pd.DataFrame(pro, columns=properties.columns)
        else:
            scaler_1 = load(open(scaler_path,'rb'))
            robust_1 = load(open(robust_std_path,'rb'))
            components = attributes[elements]
            properties = attributes[thermo_values]
            com = scaler_1.transform(components)
            pro = robust_1.transform(properties)
            comps = pd.DataFrame(com, columns=components.columns)
            props = pd.DataFrame(pro, columns=properties.columns)
            attributes = pd.concat([comps, props], axis=1)

        pred_phase = np.round(saved_model.predict(attributes),0)
        pp = pd.DataFrame(pred_phase, columns= ['AM','IM','SS','BCC$_1$','FCC$_1$','BCC$_2$','FCC$_2$'])

        
        phase_predicted =[]
        for a in pp:
            if pp[a][0]==1:
                phase_predicted.append(a)

        if 'IM' in phase_predicted:
            Phase = "PRESENT"
        else:
            Phase = "ABSENT"

        
        col1,col2 = st.columns([1,3])
        col1.markdown(
            f'<div style="display: flex; justify-content: centre;">'
            f'<span style="color: magenta; font-weight: bold; font-size: 30px; padding-left: 10px; padding-right: 10px;">'
            f'Selected HEA &#x2014;&#x2014;&#x2192;'
            f'</span>'
            f'</div>', unsafe_allow_html=True)
        col2.title(f'**:red[_{full_hea}_]**')
        col1.divider()
        col1,col2 = st.columns([1,3])
        col1.markdown(
            f'<div style="display: flex; justify-content: centre;">'
            f'<span style="color: magenta; font-weight: bold; font-size: 40px; padding-left: 10px; padding-right: 10px;">'
            f'IMC AT HEA IS &#x2014;&#x2014;&#x2192;'
            f'</span>'
            f'</div>', unsafe_allow_html=True)
        col2.title(f'**:green[_{Phase}_]**')

        prop_2 = s_col22.markdown(f"{atomic_size_difference*100:.2f} <span style='margin-left: 15px;'>&nbsp;</span> %", unsafe_allow_html=True)
        prop_2 = s_col22.markdown(f"{del_Hmix:.2f} <span style='margin-left: 15px;'>&nbsp;</span> kJ/mol", unsafe_allow_html=True)
        prop_2 = s_col22.markdown(f"{del_Smix:.2f} <span style='margin-left: 15px;'>&nbsp;</span> J/K/mol", unsafe_allow_html=True)
        prop_2 = s_col22.markdown(f"{Tm:.2f}<span style='margin-left: 15px;'>&nbsp;</span> K", unsafe_allow_html=True)
        prop_2 = s_col22.markdown(f"{Omega:.2f} <span style='margin-left: 15px;'>&nbsp;</span> ", unsafe_allow_html=True)
        prop_2 = s_col22.markdown(f"{del_Chi:.2f}<span style='margin-left: 15px;'>&nbsp;</span> ", unsafe_allow_html=True)
        prop_2 = s_col22.markdown(f"{VEC:.2f} <span style='margin-left: 15px;'>&nbsp;</span> ", unsafe_allow_html=True)

st.divider()

reset_button = st.button("RESET ALL", on_click=restart)

Citation = '''
<p style="color: white; font-size: 30px;">
If you use the result from this code or any part of this work, Please cite our paper:
<br><br>Subedi, U.; Coutinho, Y.A.; Malla, P.B.; Gyanwali, K.; Kunwar, A. Metals 2022, 12, 964.
<a href="https://doi.org/10.3390/met12060964" style="color: red; text-decoration: underline;" target="_blank">
Automatic Featurization Aided Data-Driven Method for Estimating the Presence of Intermetallic Phase in Multi-Principal Element Alloys
</a>
<a href="https://doi.org/10.3390/met12060964" style="color: green; text-decoration: underline;" target="_blank">(https://doi.org/10.3390/met12060964)</a>
</p>
'''
st.markdown(Citation, unsafe_allow_html=True)

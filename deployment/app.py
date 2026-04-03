import streamlit as st
import pandas as pd
import pickle

# load model
with open("model_best.pkl","rb") as f:
    model = pickle.load(f)


# for box square button
st.markdown("""
<style>
/* square button */
div.stButton > button {
    display: block;
    margin: auto;
    height: 60px;
    width: 200px;
    border-radius: 0px;   /* square */
    font-size: 18px;
    background-color: #4CAF50;
    color: white;
}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="main-box">', unsafe_allow_html=True)


st.markdown("""
<style>

/* LEFT LINE */
[data-testid="stAppViewContainer"]::after {
    content: "";
    position: fixed;
    top: 0;
    bottom: 0;
    left: 25%;
    width: 2px;
    background-color: white;
}

/* RIGHT LINE */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    top: 0;
    bottom: 0;
    right: 25%;
    width: 2px;
    background-color: white;
}

</style>
""", unsafe_allow_html=True)

# title
st.title(":EnergyConsumption_deployment_model:")   

st.markdown("---")
# 👉 Datetime input
datetime_input = st.text_input("Enter Datetime (YYYY-MM-DD HH:MM:SS)", "2023-01-01 12:00:00")

# convert datetime → features
dt = pd.to_datetime(datetime_input)
hour = dt.hour
day = dt.day

# numerical_col
numerical_col ={
    "Temperature":(0,50,0),
    "Humidity":(10,90,0),
    "WindSpeed":(0,10,0),
    "GeneralDiffuseFlows":(0,600,0),
    "DiffuseFlows":(0,500,0),
}

# collect input
numerical_input={}
for col, (min_val, max_val, default) in numerical_col.items():
    numerical_input[col] = st.slider(col, float(min_val), float(max_val), float(default))

# 👉 add datetime features (only 2)
numerical_input["Hour"] = hour
numerical_input["Day"] = day

input_df = pd.DataFrame([numerical_input])



# prediction
if st.button("Ennery_predict"):
    pred = model.predict(input_df)
    

    st.subheader("Energy Consumption:")

    if (pred[0]) == 3:
        st.write("PowerConsumption_Zone1:", pred[0][0])
        st.write("PowerConsumption_Zone2:", pred[0][1])
        st.write("PowerConsumption_Zone3:", pred[0][2])
    else:
        st.write(pred[0])

st.markdown("--")
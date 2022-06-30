import pandas as pd
import numpy as np
import streamlit as st
from scipy.optimize import minimize
from Predictive_Stability_Model_Functions import *
import matplotlib.pyplot as plt
import seaborn as sns

header = st.container()
dataset = st.container()
modelFeatureSelection = st.container()
modelTraining = st.container()
residuals = st.container()
# Creating containers for each section of the web page


with header:
    st.title("Predictive Stability Modelling with Models 1-10")

with dataset:
    data_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    
    if not data_file:
        st.warning('Please input a dataset on the left bar')
        st.stop()
    st.success('Thank you for inputting a dataset!')

    st.header("Dataset preview")
    df = pd.read_csv(data_file)
    st.dataframe(df)
# Creating a place for the user to upload their dataset file and then we are 
# turning that into a pandas dataframe


with modelFeatureSelection:
    st.text("Select the desired constants and variables from the left side bar")
    
    columns = df.columns
    temperature = st.sidebar.selectbox("Temperature column name", columns)
    humidity = st.sidebar.selectbox("Humidity column name", columns)
    time = st.sidebar.selectbox("Time column name", columns)
    y = st.sidebar.selectbox("Y variable column name", columns)
    
    A = st.sidebar.number_input("A", value = 11) 
    Ea = st.sidebar.number_input("Ea", value = -10000) 
    B = st.sidebar.number_input("B", value = -0.05)
    n = st.sidebar.number_input("n", value = 0.5) 
    Ap = st.sidebar.number_input("AP", value = 5)
    EaRp = st.sidebar.number_input("EaRp", value = -50000)
    Bp = st.sidebar.number_input("Bp", value = 0.05)
    
    x0 = np.array([A, Ea, B, n, Ap, EaRp, Bp])
    x0 = x0.astype(float)
    
    model_function_list = [model1_calc, model2_calc, model3_calc, model4_calc, model5_calc, model6_calc, model7_calc, model9_calc, model10_calc]
    images_list = [["Moisture Modified Arrhenius Type A","model1_calc_pic.png"], 
                    ["Moisture Modified Arrhenius Type B", "model2_calc_pic.png"], 
                    ["Diffusion Type A", "model3_calc_pic.png"], 
                    ["Diffusion Type B", "model4_calc_pic.png"], 
                    ["Shape Type A", "model5_calc_pic.png"], 
                    ["Shape Type B", "model6_calc_pic.png"], 
                    ["First Order", "model7_calc_pic.png"], 
                    ["Limit Moisture Modified Arrhenius Type A", "model9_calc_pic.png"], 
                    ["Limit Moisture Modified Arrhenius Type B", "model10_calc_pic.png"]]
    
    # images_list = ["model1_calc_pic.png", "model2_calc_pic.png", "model3_calc_pic.png", "model4_calc_pic.png", "model5_calc_pic.png", "model6_calc_pic.png", "model7_calc_pic.png", "model9_calc_pic.png", "model10_calc_pic.png"]
    
    model_image_zip = dict(zip(model_function_list, images_list))
    model_selector = st.selectbox("Select the model", model_function_list)
    st.write("Model name: " + model_image_zip[model_selector][0])
    st.image(model_image_zip[model_selector][1])
    # A selection box for the user to select the model they want to use with the image and name of the model

# Creating a few selections for the user to select what columns they want to
# align to each variable in the model and for the user to select their desired 
# starting constants that will be optimised in the model. Also outputting a image
# of the models formula  



# Making sure columns have right datatype
df[temperature] = df[temperature].astype(float)
df[humidity] = df[humidity].astype(float)
df[time] = df[time].astype(float)
df[y] = df[y].astype(float)
df = df[[temperature, humidity, time, y]]
df = df.dropna()
df = df.reset_index()


@st.cache()
def optimising_parameters(model_selector):
    optimised_parameters = minimize(model_error_squared, x0, args=(df[y], model_selector, df[temperature], df[time], df[humidity]), method='Nelder-Mead', jac=None, options={'maxiter': 99999, 'maxfev': None, 'disp': False, 'return_all': False, 'initial_simplex': None, 'xatol': 0.000001, 'fatol': 0.000001, 'adaptive': False})
    # Optimising parameters
    return optimised_parameters["x"]
# We are caching this function to optimise the performance of the web app 
# so that it stores what has been run before


with modelTraining:
    
    st.write("The temperature, humidity, time and y variable columns have been taken from the inputs on the left sidebar and a dataframe has been created. From this 4 column dataframe - all the rows with <NA> values have been omitted and the model constants have been optimised to give the following graph and constant values")
    
    y_predicted = model_selector(optimising_parameters(model_selector), df[temperature], df[time], df[humidity])
    # Predicted y values with the new optimised model parameters
    
    model_metrics = actual_vs_predicted_metrics(df[y], y_predicted)
    model_metrics_dict = {"R^2":model_metrics[0], "RMSE": model_metrics[1], "p-value":model_metrics[2]}
    
    
    # PLOT
    fig, ax = plt.subplots()
    plt.scatter(df[y], y_predicted)
    plt.title("Actual vs Predicted values")
    
    max_actual = np.amax(df[y])
    x_line = [0,max_actual]
    y_line = [0,max_actual]
    plt.plot(x_line,y_line)
    plt.ylabel("Predicted values")
    plt.xlabel("Actual values")
    plt.title("{} Actual vs Predicted Values".format(model_image_zip[model_selector][0]))
    plt.annotate("R-sqaured = {}".format(round(model_metrics_dict["R^2"], 4)), (0.5,2.5))
    plt.annotate("RMSE = {}".format(round(model_metrics_dict["RMSE"], 4)), (0.5,2.3))
    # plt.annotate("Y = {} + {}X".format(round(intercept,3),round(slope,3)), (0.5,2.1))
    
    st.pyplot(fig)
    
    st.write({"A":optimising_parameters(model_selector)[0], "Ea":optimising_parameters(model_selector)[1], "B":optimising_parameters(model_selector)[2], "n":optimising_parameters(model_selector)[3], "Ap":optimising_parameters(model_selector)[4], "EaRp":optimising_parameters(model_selector)[5], "Bp":optimising_parameters(model_selector)[6]})
    # Printing out the new set of optimised parameters
    
    
    col1, col2 = st.columns(2)
    selected_models = col1.multiselect("Select the models you want to see the metrics for", model_function_list)
    
    Model_mapping = {model1_calc: "Moisture Modified Arrhenius Type A", 
                     model2_calc: "Moisture Modified Arrhenius Type B",
                     model3_calc: "Diffusion Type A",
                     model4_calc: "Diffusion Type B",
                     model5_calc: "Shape Type A",
                     model6_calc: "Shape Type B",
                     model7_calc: "First Order",
                     # model8_calc: "",
                     model9_calc: "Limit Moisture Modified Arrhenius Type A",
                     model10_calc: "Limit Moisture Modified Arrhenius Type B"}
    
    modelForLoopSelection = col2.selectbox("Do you want a dataframe of the selected models metrics?", ["No", "Yes"])
    st.write("WARNING: Some of the models make take a long time to load their metrics e.g. model 5 and 6 will take a long time to load")
    if modelForLoopSelection == "Yes":
        metric_array = []
        for model in selected_models:
            x = actual_vs_predicted_metrics(model(optimising_parameters(model),df[temperature], df[time], df[humidity]), df[y])
            metric_array.append(x)
            
        
        model_names = pd.DataFrame(selected_models, columns = ["model"])
        model_names["model"] = model_names["model"].map(Model_mapping)
        model_metrics = pd.DataFrame(metric_array, columns = ["r_value_sqaured", "rmse", "p_value"])
        metric_df = model_names.join(model_metrics)
        # metric_df = metric_df.reset_index(drop=True)
        st.write(metric_df)

# Creating a plot of the actual values vs predicted values with the metrics of
# model on the graph. We then have an option for the user to select multiple
# models to compare by putting the different model metrics in a dataframe.
 


with residuals:
    st.write("Dataframe with actual and predicted y variables and with the residuals too")
    residual_df = pd.DataFrame({"Temperature":df[temperature], "Time":df[time], "Humidity":df[humidity], "y actual":df[y], "y predicted": y_predicted, "Residuals":df[y]-y_predicted, "Normalised Residuals": y_predicted/df[y]})
    st.write(residual_df)
    
    st.write("Pick your x, residual, colour and multiple visuals to choose which graph you want to see.")
    
    residual_columns = residual_df.columns
    col1, col2, col3, col4 = st.columns(4)
    x_column = col1.selectbox("x axis", residual_columns)
    residual_column = col2.selectbox("residual", residual_columns)
    colour_column = col3.selectbox("colour", residual_columns)
    multiples_column = col4.selectbox("multiple visuals", residual_columns)
    
    st.write("This plot is based on the model you selected above")
    fig1, ax1 = plt.subplots(nrows=1, ncols=1)
    plt.scatter(residual_df[x_column], residual_df[residual_column], c=residual_df[colour_column])
    plt.title("{} vs {} with {} as colour".format(x_column, residual_column, colour_column))
    plt.ylabel("Residuals")
    plt.legend(loc="upper left")
    plt.xlabel("{}".format(x_column))


    st.pyplot(fig1)

# Plotting the residuals against other variables to what the "good" and "bad" 
# variables are.
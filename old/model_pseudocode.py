#Assumption: crossvalidation and optimization for hyperparameters here not needed, here we code the final models

def get_data():
    '''
    Gets the data and converts it to a panda data frame. Does some first optimization for further work with the data
    '''
    #To-Do: function to load the data frame from the raw data and do some first preprocossing
    #add input parameters
    #define what needs to be returned (assumed: "df" - panda dataframe)

def preprocess_data():
    '''
    Preprocesses the data
    '''
    #To-To: write minimal preprocessing steps valid for all models in common
    #To-DO: check if one model needs some individual additional prepprocessing steps and code it
    #add input parameters
    #define what needs to be returned (assumed: "X_preprocessed", "y_preprocessed")

def train_test_split(X_preprocessed, y_preprocessed):
    '''
    Splits the already preprocessed data into train and test data
    '''
    #To-Do: create the folds
    #To-Do: Define the train_test_ratio = 0.8 (as agreed in the project group)
    #To-Do: do the split per fold
    #Return "X_preprocessed_train", "y_preprocessed_train", "X_preprocessed_test", "y_preprocessed_test"


def initialize_model(X_preprocessed_train, y_preprocessed_train):
    '''
    Initializes the model
    '''
    #To-Do: check input parameters
    #To-Do: select hyperparameters: regularizers etc.
    #To-Do: initialize the sequential model
    #To-Do: add layers to the model
    #To-Do: return the "model"

def compile_model(model):
    '''
    Compiles the model
    '''
    #To-Do: check input parameters
    #To-Do: select hyperparameters: normalizers etc.
    #To-DO: select metrics "mse" and loss "mse" (as agreed in the project group)

    #To-Do: return the model

def train_model(model):
    '''
    Trains the compiled model
    '''
    #To-Do: check input parameters
    #To-Do: select metrics,learning_rate, patience, batch_size, validation_split
    #To-Do: returns the "model" and the "history" (history is needed for further plotting in the frontend!)
    #To-Do: returns -some how - "actual_inflation"

def predict_model(model):
    '''
    Predicts for the trained model
    '''
    #To-Do: check input parameters
    #To-Do: Returns - some how - the "predicted_inflation"

def save_model(model):
    '''
    Persist trained model in GC"
    '''
    #To-Do: Choose the location where to save the model (assumed: GC)
    #To-Do: Define GC client, bucket and blob
    #To-Do: "None" to be returned


def load_model(model):
    '''
    Return a saved model (most current version)
    '''
    #To-Do: define blob, client, latest blob, latest client, latest model
    #To-Do: return "latest_model"

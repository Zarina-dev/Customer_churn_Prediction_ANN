import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import tensorflow
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation


from sklearn.metrics import confusion_matrix, classification_report

#####-----------------------------------------#################

data = pd.read_csv(r'C:\Users\admin\Desktop\python\Churn_Modelling.csv' )

print(data.columns)
# print(data.info())
# check null
def get_object_columns_unique(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            print(f'{col}: {df[col].unique()}')

# objects : surname X, geography, Gender--> map

def data_cleanig(df):
    df = df.copy()
    df = df.drop(['RowNumber','CustomerId','Surname'], axis = 1)

    return df

ds = data_cleanig(data)

print(ds.sample(5),'\n', get_object_columns_unique(ds))




##--------------------- VISUALIZATION--------------------


plt.figure(figsize=(14,8))
def hist_plot_func(col_plot_list):

    for i, col in enumerate(col_plot_list):

        plt.subplot(2,3,i+1)
        # pie for Exited
        if(col == 'Exited'):
            plt.pie(ds[col].value_counts(),explode=[0,0.2] ,
                    labels=['Stayed','Exited'],
                    shadow=True,
                    colors=['#1DB9C3','#F56FAD'])

        else:
            nos = ds[ds['Exited']==0][col]
            yess = ds[ds['Exited']==1][col]

            plt.title(f'churn, according to the {col} feature')
            sns.histplot(nos, color='#1DB9C3')
            sns.histplot(yess, color='#F56FAD')
            plt.legend(['Stayed', 'Exited'])

    plt.show()


# hist_plot_func(['Exited','Tenure','Age','IsActiveMember','EstimatedSalary','Geography'])


# # same : piv_table, display by one, groupby
ds_exited_bool = ds.copy()
ds_exited_bool['Exited_bool'] = ds_exited_bool['Exited']== 1

piv_table = pd.pivot_table(ds_exited_bool, index=['Exited'], columns= 'IsActiveMember', values='Exited_bool', aggfunc='count')
print('\n PIVOT TABLE_IsActiveMember : \n',piv_table)

# print(ds_exited_bool[ds_exited_bool['Exited']==0].IsActiveMember.value_counts())
# print(ds_exited_bool[ds_exited_bool['Exited']==1].IsActiveMember.value_counts())

# gr_isAcive = ds_exited_bool.groupby(['Exited','IsActiveMember'])['Exited_bool'].count()
# print(gr_isAcive)


###### -----------PreProcessing --------------_###############

print(get_object_columns_unique(ds))
def preprocessing(df):
    df = df.copy()
    df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
    df['Geography'] = df['Geography'].replace({'France':0, 'Spain':1,'Germany':2})

    # split
    X = df.drop('Exited',axis = 1)
    y = df.Exited

    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,random_state=1)

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = preprocessing(ds)


# ----------------------Scale-----------------------
X_train = pd.DataFrame(MinMaxScaler().fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(MinMaxScaler().fit_transform(X_test), columns=X_test.columns)

print(X_train.describe())




######----------------- Build Artific.Neural Network ---------########

# Build Network --> dense> neurons, activation function

model = Sequential([
    Dense(15, input_shape = (10,), activation = 'relu'),
    Dense(20, activation = 'relu' ),
    Dense(1, activation= 'sigmoid')
])
# print(model.summary())

model.compile(loss = 'binary_crossentropy',
              optimizer='adam',
              metrics = ['accuracy'])


model_history = model.fit(X_train, y_train, epochs = 20, validation_split=.07,verbose=2)

# --------------LOSS VISUALIZATION---------------------

print(model_history.history.keys())
# --> ['accuracy','val_accuracy','loss','val_loss']

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('Model accuracy')
plt.legend(['train','valid'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.subplot(1,2,2)
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('Loss accuracy')
plt.legend(['train','valid'])
plt.xlabel('Epochs')
plt.ylabel('loss')

plt.show()

#--------------- LOSS VISUAL END-----------------


model.evaluate(X_test,y_test)

yp = model.predict(X_test)
print(yp[:10]) # --> 0.155

y_pred = [0 if yp<0.5 else 1 for yp in yp] # binarization
print('rounded y_pred',y_pred[:10])


###########---------- Assessment of predicting value ------

print(classification_report(y_test, y_pred))

cm = tensorflow.math.confusion_matrix(labels=y_test, predictions= y_pred)

plt.figure(figsize=(7,5))
sns.heatmap(cm, annot = True, fmt = 'd')
plt.xlabel('Prediction')
plt.ylabel('Real')

plt.show()



print('*****************************  Runned!')

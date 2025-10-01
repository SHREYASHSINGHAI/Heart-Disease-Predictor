################ IMPORTS ##################################

# data analysis, manipulation and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

# importing classification models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# for model evaluation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import RocCurveDisplay

# importing the dataset 
dataset=pd.read_csv(r"dataset\HeartDiseaseTrain-Test.csv")
print("Dataset description : \n",dataset.describe)
print("\n \n \n")




###################### PROCESSING #################################

numerical_features=dataset[['age','resting_blood_pressure','cholestoral','Max_heart_rate','oldpeak']]
categorical_features=['sex', 'chest_pain_type', 'fasting_blood_sugar', 'rest_ecg', 'exercise_induced_angina', 'slope', 'vessels_colored_by_flourosopy', 'thalassemia']

# ENCODING CATEGORICAL VALUES
encoder=OneHotEncoder(sparse_output=False,handle_unknown='ignore')
encoded_categorical_cols = encoder.fit_transform(dataset[categorical_features])

numerical_df=pd.DataFrame(numerical_features)
encoded_categorical_df=pd.DataFrame(encoded_categorical_cols,columns=encoder.get_feature_names_out(categorical_features))

# #CREATING A NEW FULLY ENCODED DATASET
# encoded_df=pd.concat([numerical_df,encoded_categorical_df],axis=1)
# encoded_df['target']=dataset['target']

# #CREATING A NEW CSV FOR ENCODED DATASET
# encoded_df.to_csv(r"D:\Project1\encoded_df.csv",index=False)

df=pd.read_csv('encoded_df.csv')
df["age"]=pd.cut(df["age"],bins=[20,30,40,50,60,100])

Target_catagory_density=df["target"].value_counts()
print("\n Target catagory density: \n",Target_catagory_density)




############################ VISUALIZATION ########################################

#plotting a graph for comparision for heart disease among diferrent sex
print("\n Dataset gender distribution \n",dataset['sex'].value_counts())

sns.countplot(x='sex',data=dataset,hue="target")
plt.title("Heart disease comparision between Male & Female")
plt.xlabel("0= No disease, 1= Disease")
plt.xticks(rotation=0)
plt.show()

#plotting a graph for comparision for heart disease among diferrent age
sns.countplot(x='age',data=df,hue="target")
plt.title("Heart disease comparision among ages")
plt.xlabel("0= No disease, 1= Disease")
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.show()

#plotting correlation matrix
corr_metrix=df.corr()
fig,ax=plt.subplots(figsize=(5,5))
ax=sns.heatmap(corr_metrix,
               annot=False,
               linewidths=0.5,
               fmt=".2f",
                cmap="YlGnBu")
plt.title("Feature Correlation Metrix")
plt.show()




###################### IMPLEMENTING MODELS ###########################

#Splitting the dataset 
x=df.iloc[:,:-1]
y=df["target"]

np.random.seed(42)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

models={
"logistic_regression": LogisticRegression(),
"KNN": KNeighborsClassifier(),
"Random_forest": RandomForestClassifier()
}

def score(models,x_train,y_train,x_test,y_test):
    np.random.seed(42)
    model_scores={}
    for name,model in models.items():
        model.fit(x_train,y_train)
        model_scores[name]=model.score(x_test,y_test)

    print("\n \n BEFORE PARAMETER TUNING \n")
    print("Following are the scores : ")
    return model_scores

model_scores=score(models=models,
                   x_train=x_train,
                   y_train=y_train,
                   y_test=y_test,
                   x_test=x_test)

model_compare=pd.DataFrame(model_scores,index=["accuracy"])
print("\n Model comparision dataset before tuning : \n",model_compare)
coastal_palette = ['#011F4B', '#6497B1', '#B3CDE0']
model_compare.T.plot.bar(color=coastal_palette)
plt.grid(True,
         linewidth=0.1,
         color="black")
plt.title("Model comparision before tuning")
plt.xticks(rotation=0)
plt.show()




################### Hyperparameter Tuning ###############################
print("\n AFTER HYPERPARAMETER TUNING \n")


#KNN
print("KNN")
train_scores=[]
test_scores=[]
neighbors=range(1,21)
knn=KNeighborsClassifier()
for i in neighbors:
    knn.set_params(n_neighbors=i)
    knn.fit(x_train,y_train)
    train_scores.append(knn.score(x_train,y_train))
    test_scores.append(knn.score(x_test,y_test))
print("Train scores : \n",train_scores)
print("Test_scores : \n",test_scores)


# KNN Visualization 
plt.plot(neighbors,train_scores,label="Train scores")
plt.plot(neighbors,test_scores,label="Test scores")
plt.xticks(np.arange(1,21,1))
plt.xlabel("Number of Neighbors")
plt.ylabel("Model score")
plt.title("KNN")
plt.show()

#using Grid Search cv
print("\n USING GRID SEARCH")
knn_grid = {
    'n_neighbors': neighbors,
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']  
}
grid_search=GridSearchCV(estimator=knn,
                         param_grid=knn_grid,
                         cv=10,
                         scoring='accuracy')
grid_search.fit(x_train,y_train)
print("Best parameters found: ", grid_search.best_params_)
print("Best accuracy score: ", grid_search.best_score_)


# RANDOM FOREST    
print("\n Random Forest ") 
print("Randomized Search CV")
()
rf_grid={
    "n_estimators": np.arange(10,1000,50),
    "max_depth":[None,3,5,10],
    "min_samples_split":np.arange(2,20,2),
    "min_samples_leaf": np.arange(1,20,2)
}

np.random.seed(42)
rf_rscv=RandomizedSearchCV(RandomForestClassifier(),
                           param_distributions=rf_grid,
                           cv=5,
                           n_iter=20,
                           verbose=True)
rf_rscv.fit(x_train,y_train)
print("Best Parameters : ",rf_rscv.best_params_)
rf_rscv_acc=rf_rscv.score(x_test,y_test)
print("Accuracy",rf_rscv_acc)


#LOGISTIC REGRESSION
print("\n LOGISTIC REGRESSION")
print("USING RANDOMISED SEARCH CV")
lr_grid={
    "C":np.logspace(-40,40,200),
    "solver":["liblinear"],
}

# Applying randomised search cv in logistic regression
np.random.seed(42)
lr_rscv = RandomizedSearchCV(LogisticRegression(),
                           param_distributions=lr_grid,
                           cv=5,
                           n_iter=20,
                           verbose=True)
lr_rscv.fit(x_train,y_train)
print("Best Parameters : \n")
print(lr_rscv.best_params_)
lr_rscv_acc=lr_rscv.score(x_test,y_test)
print("Accuracy : ",lr_rscv_acc)

print("\n Grid Search CV")
lr_gs=GridSearchCV(LogisticRegression(),
                   param_grid=lr_grid,
                   cv=5,
                   verbose=True)
lr_gs.fit(x_train,y_train)
print("Best Parameters : \n")
print(lr_gs.best_params_)
lr_gs_acc=lr_gs.score(x_test,y_test)
print("Accuracy : ",lr_gs_acc)




################### EVALUATION ##################

print("LOGISTIC REGRESSION")
y_pred=lr_gs.predict(x_test)
print(y_pred)

print("ROC Curve")
RocCurveDisplay.from_estimator(lr_gs,x_test,y_test)
confusion_matrix(y_pred,y_test)
plt.title("Logistic Regression-grid search")
plt.show()

def plt_confusion_metrics(y_test,y_pred):
    fig,ax = plt.subplots(figsize=(3,3))
    ax=sns.heatmap(confusion_matrix(y_test,y_pred),
                   annot=True,
                   cbar=False)
    plt.xlabel("True Label")
    plt.ylabel("Predicted Label")

plt_confusion_metrics(y_test,y_pred)
plt.title("Logistic Regression-grid search")
plt.show()

# Classification report
print("Classification Report")
print(classification_report(y_test,y_pred))

# Calculating evaluation metrics using cross validation
# using best parameters of logistic regression

clf=LogisticRegression(C=0.09884959046625667,solver="liblinear")

cv_acc=cross_val_score(clf,x,y,cv=5,scoring="accuracy")
print("Accuracy score: ",np.mean(cv_acc))
print(' ')
cv_recall=cross_val_score(clf,x,y,cv=5,scoring="recall")
print("Recall score: ",np.mean(cv_recall))
print(' ')
cv_precision=cross_val_score(clf,x,y,cv=5,scoring="precision")
print("Precision score: ",np.mean(cv_precision))
print(' ')
cv_f1=cross_val_score(clf,x,y,cv=5,scoring="f1")
print("F1 score: ",np.mean(cv_f1))

cv_metric=pd.DataFrame({"Accuracy":[np.mean(cv_acc)],
                        "Recall":[np.mean(cv_recall)],
                        "Precision":[np.mean(cv_precision)],
                        "F1":[np.mean(cv_f1)]})
cv_metric.T.plot.bar()
plt.show()




################ FEATURE IMPORTANCE ####################################3 

clf.fit(x_train,y_train)
clf.coef_
feature_dict=dict(zip(df.columns, list(clf.coef_[0])))
print(feature_dict)
feature_df=pd.DataFrame(feature_dict,index=[0])
feature_df.T.plot.bar(title='Feature Importance',legend=False)
plt.show()

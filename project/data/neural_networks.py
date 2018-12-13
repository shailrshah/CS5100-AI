import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import time

# referenced https://scikit-learn.org/stable/modules/generated/sklearn
#fill the path to csv 
dataframe = pd.read_csv('h1b_dataset.csv')
output_mapping={'CERTIFIEDWITHDRAWN': 1, 'WITHDRAWN': 2, 'CERTIFIED': 3, 'DENIED': 4}
for val in output_mapping:
	dataframe['CASE_STATUS'] = np.where(
					dataframe['CASE_STATUS'] == val, 
					output_mapping[val], 
					dataframe['CASE_STATUS']

				   )

feature_r = [('EMPLOYER_COUNTRY', 0.0), ('FULL_TIME_POSITION', 0.0007830942186212831), ('WILLFUL_VIOLATOR', 0.000927174138450827), ('VISA_CLASS', 0.0013209130542083407), ('WAGE_UNIT_OF_PAY', 0.0014978174332684582), ('PW_UNIT_OF_PAY', 0.001926613865431721), ('TOTAL_WORKERS', 0.004696222372350042), ('DECISION_YEAR', 0.005613026591388195), ('H-1B_DEPENDENT', 0.0059109170354118045), ('PW_SOURCE', 0.007495364964875977), ('WAGE_RATE_OF_PAY_TO', 0.009982370752676516), ('PW_SOURCE_OTHER', 0.010582386258763989), ('EMPLOYER_STATE', 0.02157756310762043), ('NAICS_CODE', 0.025615064518057933), ('WORKSITE_STATE', 0.027829702920748982), ('WAGE_RATE_OF_PAY_FROM', 0.02953707528245541), ('PREVAILING_WAGE', 0.031178999525031798), ('DECISION_MONTH', 0.03499635408912244), ('WORKSITE_POSTAL_CODE', 0.045799220009636044), ('DECISION_DAY', 0.05939736127794322), ('CASE_SUBMITTED_DAY', 0.07169787988718014), ('EMPLOYER_NAME', 0.07641344119157746), ('CASE_SUBMITTED_MONTH', 0.07920810072322149), ('PW_SOURCE_YEAR', 0.11334477949151597), ('CASE_SUBMITTED_YEAR', 0.14549547967947657), ('SOC_NAME', 0.18717307761096497)]

index_r=0
for ind in xrange(index_r):
	dataframe = dataframe.drop(columns=[feature_r[ind][0]])

features = dataframe.drop(columns=['CASE_STATUS'])
feature_text = features.columns[:]
label = np.array(dataframe['CASE_STATUS'])

def labelencode(features):
	for c in features.columns:
		encoder = preprocessing.LabelEncoder()
		if type(object) == features[c].dtype: features[c] = encoder.fit_transform(features[c])
	return features

features = labelencode(features)

features.fillna(features.mean(), inplace=True)
features = np.array(features)
def main(ind,k_i,tup,iter,act):
	_file = open('ann_'+str(ind)+'.txt','w')
	start_time = time.time()
	kf = KFold(n_splits=k_i)
	for k, (train, test) in enumerate(kf.split(features, label)):
		clf = MLPClassifier(hidden_layer_sizes=tup,max_iter=iter, activation=act)
		scaler = StandardScaler()
		scaler.fit(features[train])
		features_train = scaler.transform(features[train]).astype('int') 
		features_test = scaler.transform(features[test]).astype('int')
		label_train = label[train].astype('int')
		label_test = label[test].astype('int')
		clf.fit(features_train, label_train)
		_file.write("Accuracy training"+str(k+1)+" :"+str(clf.score(features_train, label_train)*100)+'\n')	
		_file.write("Accuracy testing"+str(k+1)+" :"+str(clf.score(features_test, label_test)*100)+'\n')
		label_pred = clf.predict(features_test)
		#gives proposal results 
		_file.write(classification_report(label_test, label_pred))
		_file.write('\n')
	_file.write("--- %s seconds ---\n" % (time.time() - start_time))
	_file.write("------:K-fold: " + str(k_i) + " :hidden: " + str(tup) + " :iteration: " + str(iter) + " :activationFunction: " + str(act) + '----------\n')
	_file.close()

actLogistic = 'logistic'
actTanh = 'tanh'
actRelu = 'relu'

simulation = [(3,(13,13,13),500,actRelu), (3,(13,13,13),250,actRelu), (5,(13,13,13),250,actRelu), (5,(13),250,actRelu), (3,(13),250,actRelu), (3,(13,13,13,13,13),250,actRelu), (3,(5,5,5),250,actRelu), (3,(20,20,20),250,actRelu), (5,(13,13,13),250,actLogistic), (3,(13,13,13),250,actLogistic), (5,(13,13,13),250,actTanh), (3,(13,13,13),250,actTanh)]
ind=1
for entry in simulation:
	main(ind,entry[0],entry[1],entry[2],entry[3])
	ind+=1
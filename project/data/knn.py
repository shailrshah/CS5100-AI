import pandas as pd
import numpy as np
from sklearn.neighbors  import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler  
import time
from sklearn.model_selection import KFold

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
#gets the dimensions of the dataframe (528134, 27)
#print dataframe.shape
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
def main(ind,k_i,n):
	_file = open('knn_'+str(ind)+'.txt','w')
	start_time = time.time()
	kf = KFold(n_splits=k_i)
	for k, (train, test) in enumerate(kf.split(features, label)):
		clf = KNeighborsClassifier(n_jobs=20, n_neighbors=n)
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
	_file.write("------:K-fold: " + str(k_i) + " :neighbors: " + str(n) + '----------\n')
	_file.close()

simulation = [(3,3),(3,1),(3,5)]
ind=1
for entry in simulation:
	main(ind,entry[0],entry[1])
	ind+=1
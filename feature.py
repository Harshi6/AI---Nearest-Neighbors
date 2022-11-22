import csv
import numpy as np
import pandas as pd
#used for reading files easier
import math
#used for square root
import copy
#copy used for deepcopy
import sys
#used for INT_MAX
import time
#used for keeping track of runtime

def main():
	print("Welcome to Harshi Doddapaneni\'s Feature Selection Algorithm!")
	#Get the test file from user input
	IN_FILE = input("Type in the name of the file to test: ")
	#reading the file
	test = open(IN_FILE, 'r')
	read = csv.reader(test,delimiter=' ',skipinitialspace=True)
	#find number of features
	features_count = len(next(read)) 
	#find the number of instances
	inst = pd.read_csv(IN_FILE)
	instances_count= len(inst) + 1
	#ask user which algorithm they would like to use    
	question = "\nType the number of the algorithm you want to run.\n"
	question += "1.Forward Selection\n"
	question += "2.Backward Elimination"
	print(question)
	#get user input
	alg = int(input())
	fc = features_count - 1
	#print number of features and instances in the dataset
	print("This dataset has {} features (not including the class attribute), with {} instances\n".format(fc,instances_count))
	#getting all features
	fts = []
	for i in range(1, features_count):
		fts.append(i)
	#getting accuracy
	df = pd.read_fwf(IN_FILE, header=None)
	data = df.copy(deep=True)[:-1]
	acc = leave_one_out(features_count,data,fts)
	print("Running nearest neighbor with all {} features, using \"leaving-one-out\" evalutation, I get an accuracy of {}%\n".format(fc, acc))
	#follow through accordingly based on user answer
	#Begin searching! 
	if alg == 1:
		#start timer as search is starting
		start = time.time()
		forward_selection(features_count,IN_FILE,start)
	if alg == 2:
		#start timer as search is starting
		start = time.time()
		backward_elimination(features_count,IN_FILE,start)

#start at the top, then we go through every feature and select best one. Then we look at subsets and choose the following best subset.
#important to keep track of a main main accuaracy score and a sub_accuracy score. main accuracy score will be used to find the 
#best feature set out of all of them. Sub_accuracy score will be used to find the best accuracy score within that current level and its options
def forward_selection(features_count, IN_FILE, start):
	print("Beginning search.\n")
	df = pd.read_fwf(IN_FILE, header=None)
	#need for caculating accuracy
	data = df.copy(deep=True)[:-1]
	# tracking answer set
	ans = []
	#tracking the features that we have already seen
	fseen = []
	#top acc
	global_acc = 0
	
	for i in range(1, features_count):
		#tracking local accuracy in the curr level
		local_acc = 0
		#values we will be appending based on accuracy comparisons
		add = 0
		add2 = 0
		#bool to check if triggered
		check1 = False
		check2 = False
		for features in range(1, features_count):
			if features in fseen:
				continue
			# we do not need to compare the same one. Ex. If 4 is already in set, we don't need to add it and compare
			#therefore we make sure it is not already in subset
			if features not in fseen:
				#need to deepcopy when iterating and updating changes
				dummy = copy.deepcopy(fseen)
				dummy.append(features)
				#lets get current accuracy with current set and its features
				curr_acc = leave_one_out(features_count,data,dummy)
				#checking for new max acc and updating accordingly, both global and local accuracy
				if curr_acc > local_acc:
					local_acc = curr_acc
					check1 = True
					add = features
				if curr_acc > global_acc:
					global_acc = curr_acc
					check2 = True
					add2 = features
				print("\tUsing feature(s) {} accuracy is {}%".format(dummy,curr_acc))
		#check to see if top accuracy still greater. If so, we give warning message. If it got replaced, no warning message.		
		if check2 == False:
			print("\n(Warning, Accuracy has decreased! Continuing search in case of local maxima)")
			fseen.append(add)
			print("Feature set {} was best, accuracy is {}%\n".format(fseen, local_acc))
		if check2 == True:
			#only append in ans set when top accuracy changed. Otherwise we only append to local set.
			ans.append(add2)
			fseen.append(add2)
			print("\nFeature set {} was best, accuracy is {}%\n".format(fseen, global_acc))
	#return answer
	print("Finished search. The best feature subset is {}, which has an accuracy of {}%".format(ans, global_acc))
	#stop timer
	end = time.time()
	t = end - start
	#precision
	ti = round(t,2)
	#return runtime
	print("Time took: {} seconds".format(ti))

#backward is similar to forward search but this time we start from the opposite side.
#in forward, you start with nothing and you compare the best options and append them to the set
#this time in backward, we will start from the end. So now we start with all of the features in the set
#then using accuracy comparisons, we can decide which features to throw away or remove from the set
#we will keep track of the main acc (all time best) and sub accuracy for according level
def backward_elimination(features_count, IN_FILE, start):
	print("Beginning search.\n")
	df = pd.read_fwf(IN_FILE, header=None)
	#need for caculating accuracy
	data = df.copy(deep=True)[:-1]
	#answer set
	ans = []
	#top acc
	global_acc = 0
	#seen set
	fseen = []
	#need to get all features loaded in, for both the main answer and subset
	for feature in range(1, features_count):
		fseen.append(feature)
		ans.append(feature)

	for i in range(1, features_count):
		#local accuracy
		local_acc = 0
		#bool to check if triggered
		check1 = False
		check2 = False
		#values we will be removing based on accuracy comparisons
		remove = 0
		remove2 = 0
		for feature in range(1, features_count):
			#if feature not in subset, then we can simply skip because it is not possible to remove it. 
			#only need to go through iteration when it is part of the subset
			if feature not in fseen:
				continue
			if feature in fseen:
				#need to deepcopy when iterating and updating changes
				dummy = copy.deepcopy(fseen)
				dummy.remove(feature)
				#lets get current accuracy with current set and its features. We then use this value to compare
				curr_acc = leave_one_out(features_count, data, dummy)
				#updating the sub and main accuracy 
				if curr_acc > local_acc:
					local_acc = curr_acc
					check1 = True
					remove = feature
				if curr_acc > global_acc:
					global_acc = curr_acc
					check2 = True
					remove2 = feature
				print("\tUsing feature(s) {} accuracy is {}%".format(dummy,curr_acc))
		#check to see if top accuracy still greater. If so, we give warning message. If it got replaced, no warning message.
		if check2 == False:
			print("\n(Warning, Accuracy has decreased! Continuing search in case of local maxima)")
			fseen.remove(remove)
			print("Feature set {} was best, accuracy is {}%\n".format(fseen, local_acc))
		if check2 == True:
			#only remove from main set when top accuracy changed. Otherwise we only append to local set.
			ans.remove(remove2)
			fseen.remove(remove2)
			print("\nFeature set {} was best, accuracy is {}%\n".format(fseen, global_acc))
	#return answer
	print("Finished search. The best feature subset is {}, which has an accuracy of {}%".format(ans, global_acc))
	#stop timer
	end = time.time()
	t = end - start
	#precision
	ti = round(t,2)
	#return runtime
	print("Time took: {} seconds".format(ti))

def leave_one_out(ft, data, curr):
	#rows
	size = len(data.index)
	#correct answers counter
	valid = 0
	#just converting
	new = data.copy(deep=True)
	nd = new.to_numpy(dtype='float', na_value=np.nan)
	#setting them to 0 since we dont look at them
	for f in range(1, ft):
		if f not in curr:
			nd[:, f] = 0
	#checking distances
	for r in range(0, size):
		#classifying process 
		checking = nd[r][1:]
		label = nd[r][0]
		nn_d = sys.maxsize #nearest neighbor distance, setting to INT_MAX
		loc = sys.maxsize #tracking location, setting to INT_MAX
		
		for i in range(0, size):
			#making sure we are not checking same row
			if i == r:
				continue
			#different row
			if i != r:
				#calculating euclidean distance
				dist = math.sqrt(sum(np.power((checking - nd[i][1:]), 2)))
				#updating the least dist, and keeping track of the corresponding instance loc
				if dist < nn_d:
					loc = i
					nn_label = nd[loc][0] #getting class label
					nn_d = dist
					
		# we are checking to see if they are classified correctly based on class
		if label == nn_label:
			valid += 1
	#accuracy = number of correct classifications / number of instances in our database
	acc = valid / size
	p = acc * 100 #converting to percentage
	acc = round(p,1) #precision
	return acc

#start	
main()

import csv
import numpy as np
import pandas as pd



def main():
	print("Welcome to Harshi Doddapaneni\'s Feature Selection Algorithm!")

	IN_FILE = input("Type in the name of the file to test: ")
	#reading the file
	test = open(IN_FILE, 'r')
	read = csv.reader(test,delimiter=' ',skipinitialspace=True)

	features_count = len(next(read)) 
	#find the number of instances
	inst = pd.read_csv(IN_FILE)
	instances_count= len(inst) + 1

	question = "\nType the number of the algorithm you want to run.\n"
	question += "1.Forward Selection\n"
	question += "2.Backward Elimination"
	print(question)

	alg = int(input())
	fc = features_count - 1

	print("This dataset has {} features (not including the class attribute), with {} instances\n".format(fc,instances_count))

	fts = []
	for i in range(1, features_count):
		fts.append(i)
	df = pd.read_fwf(IN_FILE, header=None)
	data = df.copy(deep=True)[:-1]
	acc = leave_one_out(features_count,data,fts)
	print("Running nearest neighbor with all {} features, using \"leaving-one-out\" evalutation, I get an accuracy of {}%\n".format(fc, acc))


	if alg == 1:

		start = time.time()
		forward_selection(features_count,IN_FILE,start)
	if alg == 2:

		start = time.time()
		backward_elimination(features_count,IN_FILE,start)

def forward_selection(features_count, IN_FILE, start):
	print("Beginning search.\n")
	df = pd.read_fwf(IN_FILE, header=None)
	data = df.copy(deep=True)[:-1]
	ans = []
	fseen = []
	acc = 0
	for i in range(1, features_count):

		local_acc = 0

		add = 0
		
		
		for features in range(1, features_count):
			
			if features not in fseen:
			
				dummy = copy.deepcopy(fseen)
				dummy.append(features)
	
				acc = leave_one_out(features_count,data,dummy)
			
				if acc > local_acc:
					local_acc = acc
					add = features
				if acc > global_acc:
					global_acc = acc
					add = features
					
def backward_elimination(features_count, IN_FILE, start):
	print("Beginning search.\n")
	df = pd.read_fwf(IN_FILE, header=None)
	data = df.copy(deep=True)[:-1]
	ans = []
	fseen = []
	acc = 0
	for feature in range(1, features_count):
		fseen.append(feature)
		ans.append(feature)
	for i in range(1, features_count):

		local_acc = 0

		remove = 0
		
		
		for features in range(1, features_count):
			
			if features not in fseen:
			
				dummy = copy.deepcopy(fseen)
				dummy.append(features)
	
				acc = leave_one_out(features_count,data,dummy)
			
				if acc > local_acc:
					local_acc = acc
					add = features
				if acc > global_acc:
					global_acc = acc
					add = 

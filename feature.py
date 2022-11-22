def main():
	print("Welcome to Harshi Doddapaneni\'s Feature Selection Algorithm!")

	IN_FILE = input("Type in the name of the file to test: ")
	test = open(IN_FILE, 'r')
	read = csv.reader(test,delimiter=' ',skipinitialspace=True)

	features_count = len(next(read)) 

	inst = pd.read_csv(IN_FILE)
	instances_count= len(inst) + 1

	question = "\nType the number of the algorithm you want to run.\n"
	question += "1.Forward Selection\n"
	question += "2.Backward Elimination"
	print(question)

	alg = int(input())
	fc = features_count - 1

	print("This dataset has {} features (not including the class attribute), with {} instances\n".format(fc,instances_count))


	if alg == 1:

		forward_selection(features_count,IN_FILE,start)
	if alg == 2:
	
		backward_elimination(features_count,IN_FILE,start)

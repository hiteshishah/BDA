import numpy as np

data = np.genfromtxt("HW_05C_DecTree_TESTING__FOR_STUDENTS__v540.csv", delimiter=",", dtype="unicode")
attributes = data[0]
data = data[1:].astype(np.float)
classes = []
for line in data:
	if line[4] <= 2.85:
		if line[2] <= -0.06:
			classes.append('Whippet')
		else:
			if line[3] <= -0.05:
				classes.append('Greyhound')
			else:
				if line[5] <= 0.09:
					classes.append('Whippet')
				else:
					if line[0] <= -0.18:
						classes.append('Whippet')
					else:
						if line[1] <= -0.17:
							classes.append('Greyhound')
						else:
							classes.append('Whippet')
	else:
		if line[3] <= 4.0:
			if line[1] <= 8.14:
				if line[5] <= 8.14:
					if line[2] <= 8.06:
						if line[0] <= 8.17:
							classes.append('Greyhound')
						else:
							classes.append('Whippet')
					else:
						classes.append('Greyhound')
				else:
					classes.append('Greyhound')
			else:
				classes.append('Greyhound')
		else:
			if line[5] <= 4.02:
				classes.append('Whippet')
			else:
				if line[0] <= 8.19:
					if line[1] <= -0.19:
						classes.append('Greyhound')
					else:
						if line[2] <= -0.15:
							classes.append('Whippet')
						else:
							classes.append('Greyhound')
				else:
					classes.append('Greyhound')

np.savetxt("HW05_Shah_Hiteshi_MyClassifications.csv", classes, fmt="%s", delimiter=",", header="Class", comments="")
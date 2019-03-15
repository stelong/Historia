def divide_et_impera(name_in):
	with open(name_in + '.txt', 'r') as f:
		lines = f.readlines()
		for i in range(4):
			with open(name_in + '_' + str(i+1) + '.txt', 'w') as fi:
				for c, line in enumerate(lines):
					if c // 200 == i:
						fi.write(line)
	return

def append_to_header(rat, ca_traces, number):
	with open('utils/headers/' + rat + '/MeshFlatBase.in', 'r') as f:
		with open('MeshFlatBase_' + str(number) + '.in', 'w') as fh:
			for line in f:
				fh.write(line)
			with open(ca_traces + '_' + str(number) + '.txt', 'r') as fc:
				for line in fc:
					fh.write(line)
	return

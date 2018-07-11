import pickle as pck
import pandas as pd
import numpy as np
import h5py
import zipfile

problem = "easy"


###############################################################################################################################################
if problem == "ifpen":

	filtrations = ["density2", "dtm1", "X", "Y", "Z"]
	features = ["length2D", "void2D", "void3D", "area", "dtm1"]
	train_file = h5py.File("../../../../Documents/datasets/ifpen/train_diag.hdf5", "a")
	test_file  = h5py.File("../../../../Documents/datasets/ifpen/test_diag.hdf5",  "a")

	path_to_features = "../../../../Documents/datasets/ifpen/features/"
	train_F, test_F = [], []
	for fn in features:

		train_fn, test_fn = np.array(pck.load(open(path_to_features + fn + ".dat", "rb"))), np.array(pck.load(open(path_to_features + fn + "_test.dat", "rb")))
		if len(train_fn.shape) == 1:
			train_fn, test_fn = np.transpose(np.array([train_fn])), np.transpose(np.array([test_fn]))
		train_F.append(train_fn)
		test_F.append(test_fn)

	train_F = np.concatenate(train_F, 1)
	test_F  = np.concatenate(test_F, 1)

	test_df = pd.DataFrame(test_F)
	train_df = pd.concat([pd.read_csv("../../../../Documents/datasets/ifpen/targets.csv", sep = ";").drop("ID", 1), pd.DataFrame(train_F)], axis = 1)
	train_df.to_csv("../../../../Documents/datasets/ifpen/train.csv")
	test_df.to_csv("../../../../Documents/datasets/ifpen/test.csv")


	path_to_diag = "../../../../Documents/datasets/ifpen/diagrams/"
	for f in filtrations:

		train_file.create_group(f)
		test_file.create_group(f)
		train_group_f, test_group_f = train_file[f], test_file[f]

		train_group_f.create_group("0")
		train_group_f.create_group("1")
		train_group_f.create_group("2")
		test_group_f.create_group("0")
		test_group_f.create_group("1")
		test_group_f.create_group("2")

		train_group0, train_group1, train_group2 = train_group_f["0"], train_group_f["1"], train_group_f["2"]
		test_group0,  test_group1,  test_group2  = test_group_f["0"],  test_group_f["1"],  test_group_f["2"]
		if f == "density2":
			name_diag = "SetDensity2_"
		if f == "dtm1":
			name_diag = "DTM1_"
		if f == "X" or f == "Y" or f == "Z":
			name_diag = ""

		for i in range(0, 400):

			diag = pck.load(open("../../../../Documents/datasets/ifpen/diagrams/" + f + "/dgm_" + name_diag + str(i) + ".dat", "rb"))
			diag0, diag1, diag2, num_pts_in_diag = [], [], [], len(diag)
			for j in range(num_pts_in_diag):
				if diag[j][0] == 0:
					diag0.append(diag[j][1])
				if diag[j][0] == 1:
					diag1.append(diag[j][1])
				if diag[j][0] == 2:
					diag2.append(diag[j][1])
			train_group0.create_dataset(name = str(i), data = diag0)
			train_group1.create_dataset(name = str(i), data = diag1)
			train_group2.create_dataset(name = str(i), data = diag2)

		for i in range(400, 512):

			diag = pck.load(open("../../../../Documents/datasets/ifpen/diagrams/" + f + "/dgm_" + name_diag + str(i) + ".dat", "rb"))
			diag0, diag1, diag2, num_pts_in_diag = [], [], [], len(diag)
			for j in range(num_pts_in_diag):
				if diag[j][0] == 0:
					diag0.append(diag[j][1])
				if diag[j][0] == 1:
					diag1.append(diag[j][1])
				if diag[j][0] == 2:
					diag2.append(diag[j][1])
			test_group0.create_dataset(name = str(i-400), data = diag0)
			test_group1.create_dataset(name = str(i-400), data = diag1)
			test_group2.create_dataset(name = str(i-400), data = diag2)



###############################################################################################################################################
if problem == "airplane":

	train_file = h5py.File("../../../../Documents/datasets/airplane/train_diag.hdf5", "a")
	train_file.create_group("geodesic")
	train_group_f = train_file["geodesic"]

	train_group_f.create_group("0")
	train_group_f.create_group("1")
	train_group_f.create_group("2")

	train_group0, train_group1, train_group2 = train_group_f["0"], train_group_f["1"], train_group_f["2"]

	idx, lab = 0, []
	for i in range(61, 80):

		air_zip = zipfile.ZipFile("../../../../Documents/datasets/airplane/" + str(i) + "-OrdPD.zip")
		labels = np.loadtxt("../../../../Documents/datasets/airplane/" + str(i) + ".lab")

		for j in range(1, 3000, 10):
			air_zip.extract("OrdPD-" + str(i) + "-" + str(j), "../../../../Documents/datasets/airplane/")
			diag = np.loadtxt("../../../../Documents/datasets/airplane/OrdPD-" + str(i) + "-" + str(j))
			diag0, diag1, diag2, num_pts_in_diag = [], [], [], diag.shape[0]
			for k in range(num_pts_in_diag):
				if diag[k-1, 2] == 0:
					diag0.append(diag[k-1,0:2])
				if diag[k-1, 2] == 1:
					diag1.append(diag[k-1,0:2])
				if diag[k-1, 2] == 2:
					diag2.append(diag[k-1,0:2])
			lab.append(labels[j-1])
			train_group0.create_dataset(name = str(idx), data = diag0)
			train_group1.create_dataset(name = str(idx), data = diag1)
			train_group2.create_dataset(name = str(idx), data = diag2)
			idx += 1

	df = pd.DataFrame(lab, columns = ["part"])
	df.to_csv("../../../../Documents/datasets/airplane/train.csv")


###############################################################################################################################################
if problem == "smartphone":

	train_file = h5py.File("../../../../Documents/datasets/smartphone/train_diag.hdf5", "a")
	train_file.create_group("accelerometer")
	train_group_f = train_file["accelerometer"]

	train_group_f.create_group("0")
	train_group_f.create_group("1")

	train_group0, train_group1 = train_group_f["0"], train_group_f["1"]

	F, lab = pck.load(open("../../../../Documents/datasets/smartphone/data_acc_alpha", "rb")), []

	for i in range(0, 100):
		lab.append(0)
		diag = F[i]
		num_pts, diag0, diag1 = len(diag), [], []
		for j in range(num_pts):
			if(diag[j][0] == 0):
				diag0.append(  [diag[j][1][0], diag[j][1][1]]  )
			if(diag[j][0] == 1):
				diag1.append(  [diag[j][1][0], diag[j][1][1]]  )
		train_group0.create_dataset(name = str(i), data = diag0)
		train_group1.create_dataset(name = str(i), data = diag1)

	for i in range(100, 200):
		lab.append(1)
		diag = F[i]
		num_pts, diag0, diag1 = len(diag), [], []
		for j in range(num_pts):
			if(diag[j][0] == 0):
				diag0.append(  [diag[j][1][0], diag[j][1][1]]  )
			if(diag[j][0] == 1):
				diag1.append(  [diag[j][1][0], diag[j][1][1]]  )
		train_group0.create_dataset(name = str(i), data = diag0)
		train_group1.create_dataset(name = str(i), data = diag1)


	for i in range(200, 300):
		lab.append(2)
		diag = F[i]
		num_pts, diag0, diag1 = len(diag), [], []
		for j in range(num_pts):
			if(diag[j][0] == 0):
				diag0.append(  [diag[j][1][0], diag[j][1][1]]  )
			if(diag[j][0] == 1):
				diag1.append(  [diag[j][1][0], diag[j][1][1]]  )
		train_group0.create_dataset(name = str(i), data = diag0)
		train_group1.create_dataset(name = str(i), data = diag1)

	df = pd.DataFrame(lab, columns = ["walker"])
	df.to_csv("../../../../Documents/datasets/smartphone/train.csv")

###############################################################################################################################################
def z_bridge(i):
	if 1 <= i <= 9:
		nz = "0000"
	if 10 <= i <= 99:
		nz = "000"
	if 100 <= i <= 999:
		nz = "00"
	if 1000 <= i <= 9999:
		nz = "0"
	if 10000 <= i <= 14980:
		nz = ""
	return nz

def man_label(i):
	lab = -1
	if 1 <= i <= 2000:
		lab = 0
	if 4000 <= i <= 6000:
		lab = 1
	if 8000 <= i <= 9000:
		lab = 2
	if 10000 <= i <= 11000:
		lab = 3
	return lab

if problem == "bridge":

	train_file_dg = h5py.File("../../../../Documents/datasets/bridge/s3/train_diag.hdf5", "a")
	train_file_dg.create_group("fdtm")
	train_group_f = train_file_dg["fdtm"]
	train_group_f.create_group("0")
	train_group_f.create_group("1")
	#train_group_f.create_group("2")
	train_group0, train_group1 = train_group_f["0"], train_group_f["1"]
	#train_group2 = train_group_f["2"]

	path0 = "../../../../Documents/datasets/bridge/f-dtm/f-dtm0/"
	path1 = "../../../../Documents/datasets/bridge/f-dtm/f-dtm1/"
	path2 = "../../../../Documents/datasets/bridge/f-dtm/f-dtm2/"

	#train_file_pc = h5py.File("../../../../Documents/datasets/bridge/train_pc.hdf5", "a")

	lab = []

	count = 0

	for i in range(1,14981,3):
		cloud = np.loadtxt("../../../../Documents/datasets/bridge/pc/Vib-No3-Y-"+ z_bridge(i) + str(i))
		if np.sum(cloud) != 0 and man_label(i) >= 0:
			D0,D1 = np.loadtxt(path0 + "w0-" + str(i)), np.loadtxt(path1 + "w1-" + str(i))
			#D2 = np.loadtxt(path2 + "w2-" + str(i))
			train_group0.create_dataset(name = str(count), data = D0)
			train_group1.create_dataset(name = str(count), data = D1)
			#train_group2.create_dataset(name = str(count), data = D2)
			#train_file_pc.create_dataset(name = str(count), data = cloud)
			count += 1
			lab.append(man_label(i))

	#df = pd.DataFrame(lab, columns = ["degree"])
	#df.to_csv("../../../../Documents/datasets/bridge/train_s3.csv")



###############################################################################################################################################
if problem == "ltm":

	train_file_dg = h5py.File("../../../../Documents/datasets/ltm/train_diag.hdf5", "a")
	train_file_dg.create_group("alpha")
	train_group_f = train_file_dg["alpha"]
	train_group_f.create_group("0")
	train_group_f.create_group("1")
	train_group0, train_group1 = train_group_f["0"], train_group_f["1"]

	path0 = "../../../../Documents/datasets/ltm/r2.5/"
	path1 = "../../../../Documents/datasets/ltm/r3.5/"
	path2 = "../../../../Documents/datasets/ltm/r4/"
	path3 = "../../../../Documents/datasets/ltm/r4.1/"
	path4 = "../../../../Documents/datasets/ltm/r4.3/"

	lab = []

	count = 0

	for i in range(1,51):
		D0, D1, D2, D3, D4 = np.loadtxt(path0 + "OrdPD-2.5-" + str(i)), np.loadtxt(path1 + "OrdPD-3.5-" + str(i)), np.loadtxt(path2 + "OrdPD-4-" + str(i)), np.loadtxt(path3 + "OrdPD-4.1-" + str(i)), np.loadtxt(path4 + "OrdPD-4.3-" + str(i))

		lab.append(0)
                num_pts, diag0, diag1 = D0.shape[0], [], []
                for j in range(num_pts):
                        if(D0[j,2] == 0):
                                diag0.append(D0[j,:2])
                        if(D0[j,2] == 1):
                                diag1.append(D0[j,:2])
                train_group0.create_dataset(name = str(count), data = diag0)
                train_group1.create_dataset(name = str(count), data = diag1)
		count += 1

		lab.append(1)
                num_pts, diag0, diag1 = D1.shape[0], [], []
                for j in range(num_pts):
                        if(D1[j,2] == 0):
                                diag0.append(D1[j,:2])
                        if(D1[j,2] == 1):
                                diag1.append(D1[j,:2])
                train_group0.create_dataset(name = str(count), data = diag0)
                train_group1.create_dataset(name = str(count), data = diag1)
		count += 1

		lab.append(2)
                num_pts, diag0, diag1 = D2.shape[0], [], []
                for j in range(num_pts):
                        if(D2[j,2] == 0):
                                diag0.append(D2[j,:2])
                        if(D2[j,2] == 1):
                                diag1.append(D2[j,:2])
                train_group0.create_dataset(name = str(count), data = diag0)
                train_group1.create_dataset(name = str(count), data = diag1)
		count += 1

		lab.append(3)
                num_pts, diag0, diag1 = D3.shape[0], [], []
                for j in range(num_pts):
                        if(D3[j,2] == 0):
                                diag0.append(D3[j,:2])
                        if(D3[j,2] == 1):
                                diag1.append(D3[j,:2])
                train_group0.create_dataset(name = str(count), data = diag0)
                train_group1.create_dataset(name = str(count), data = diag1)
		count += 1

		lab.append(4)
                num_pts, diag0, diag1 = D4.shape[0], [], []
                for j in range(num_pts):
                        if(D4[j,2] == 0):
                                diag0.append(D4[j,:2])
                        if(D4[j,2] == 1):
                                diag1.append(D4[j,:2])
                train_group0.create_dataset(name = str(count), data = diag0)
                train_group1.create_dataset(name = str(count), data = diag1)
		count += 1

	df = pd.DataFrame(lab, columns = ["parameter"])
	df.to_csv("../../../../Documents/datasets/ltm/train.csv")


###############################################################################################################################################
if problem == "reddit":

	path = "../../../../Documents/datasets/reddit/degree.hdf5"
        D = h5py.File(path)

	train_file_dg = h5py.File("../../../../Documents/datasets/reddit/train_diag.hdf5", "a")
	train_file_dg.create_group("degree")
	train_group_f = train_file_dg["degree"]
	train_group_f.create_group("0")
	train_group_f.create_group("1")
	train_group0, train_group1 = train_group_f["0"], train_group_f["1"]

	lab = []
	count, label = 0, 0

	for i in range(0,1000):
		D0  = np.array(D["data_views"]["DegreeVertexFiltration_dim_0"]["1"][str(i)])
		D0e = np.array(D["data_views"]["DegreeVertexFiltration_dim_0_essential"]["1"][str(i)])
                D1e = np.array(D["data_views"]["DegreeVertexFiltration_dim_1_essential"]["1"][str(i)])
		if D1e.shape[0] > 0:
			D1e[:,1] = np.full([D1e.shape[0]], np.inf)
                else:
                        D1e = np.reshape(D1e,[0,2])
		D0e[:,1] = np.full([D0e.shape[0]], np.inf)
		train_group0.create_dataset(name = str(count), data = np.concatenate([D0e,D0], 0))
                train_group1.create_dataset(name = str(count), data = D1e)
		count += 1
                lab.append(label)

	label += 1
	for i in range(1000,2000):
		D0  = np.array(D["data_views"]["DegreeVertexFiltration_dim_0"]["4"][str(i)])
		D0e = np.array(D["data_views"]["DegreeVertexFiltration_dim_0_essential"]["4"][str(i)])
                D1e = np.array(D["data_views"]["DegreeVertexFiltration_dim_1_essential"]["4"][str(i)])
		if D1e.shape[0] > 0:
			D1e[:,1] = np.full([D1e.shape[0]], np.inf)
                else:
                        D1e = np.reshape(D1e,[0,2])
		D0e[:,1] = np.full([D0e.shape[0]], np.inf)
		train_group0.create_dataset(name = str(count), data = np.concatenate([D0e,D0], 0))
                train_group1.create_dataset(name = str(count), data = D1e)
		count += 1
                lab.append(label)

	label += 1
	for i in range(2000,3000):
		D0  = np.array(D["data_views"]["DegreeVertexFiltration_dim_0"]["3"][str(i)])
		D0e = np.array(D["data_views"]["DegreeVertexFiltration_dim_0_essential"]["3"][str(i)])
                D1e = np.array(D["data_views"]["DegreeVertexFiltration_dim_1_essential"]["3"][str(i)])
		if D1e.shape[0] > 0:
			D1e[:,1] = np.full([D1e.shape[0]], np.inf)
                else:
                        D1e = np.reshape(D1e,[0,2])
		D0e[:,1] = np.full([D0e.shape[0]], np.inf)
		train_group0.create_dataset(name = str(count), data = np.concatenate([D0e,D0], 0))
                train_group1.create_dataset(name = str(count), data = D1e)
		count += 1
                lab.append(label)

	label += 1
	for i in range(3000,3999):
		D0  = np.array(D["data_views"]["DegreeVertexFiltration_dim_0"]["5"][str(i)])
		D0e = np.array(D["data_views"]["DegreeVertexFiltration_dim_0_essential"]["5"][str(i)])
                D1e = np.array(D["data_views"]["DegreeVertexFiltration_dim_1_essential"]["5"][str(i)])
		if D1e.shape[0] > 0:
			D1e[:,1] = np.full([D1e.shape[0]], np.inf)
                else:
                        D1e = np.reshape(D1e,[0,2])
		D0e[:,1] = np.full([D0e.shape[0]], np.inf)
		train_group0.create_dataset(name = str(count), data = np.concatenate([D0e,D0], 0))
                train_group1.create_dataset(name = str(count), data = D1e)
		count += 1
                lab.append(label)

	label += 1
	for i in range(3999,4999):
		D0  = np.array(D["data_views"]["DegreeVertexFiltration_dim_0"]["2"][str(i)])
		D0e = np.array(D["data_views"]["DegreeVertexFiltration_dim_0_essential"]["2"][str(i)])
                D1e = np.array(D["data_views"]["DegreeVertexFiltration_dim_1_essential"]["2"][str(i)])
		if D1e.shape[0] > 0:
			D1e[:,1] = np.full([D1e.shape[0]], np.inf)
                else:
                        D1e = np.reshape(D1e,[0,2])
		D0e[:,1] = np.full([D0e.shape[0]], np.inf)
		train_group0.create_dataset(name = str(count), data = np.concatenate([D0e,D0], 0))
                train_group1.create_dataset(name = str(count), data = D1e)
		count += 1
                lab.append(label)

	df = pd.DataFrame(lab, columns = ["topic"])
	df.to_csv("../../../../Documents/datasets/reddit/train.csv")



###############################################################################################################################################
if problem == "easy":

	train_file_dg = h5py.File("../../../../Documents/datasets/easy/train_diag.hdf5", "a")
	train_file_dg.create_group("handmade")
	train_group_f = train_file_dg["handmade"]
	train_group_f.create_group("0")
	train_group0 = train_group_f["0"]

	lab = []
	count, label = 0, 0

	for i in range(100):
		diag = np.random.normal(loc = [0.0, 5.0], scale = [0.1, 0.1], size = [50,2])
		train_group0.create_dataset(name = str(count), data = diag)
		count += 1
		lab.append(label)

	label += 1
	for i in range(100):
		diag = np.random.normal(loc = [0.0, 10.0], scale = [0.1, 0.1], size = [50,2])
		train_group0.create_dataset(name = str(count), data = diag)
		count += 1
		lab.append(label)

	label += 1
	for i in range(100):
		diag = np.random.normal(loc = [5.0, 10.0], scale = [0.1, 0.1], size = [50,2])
		train_group0.create_dataset(name = str(count), data = diag)
		count += 1
		lab.append(label)

	df = pd.DataFrame(lab, columns = ["group"])
	df.to_csv("../../../../Documents/datasets/easy/train.csv")


from dependencies import *

def plot_train_image_with_annotation(index):
	mpii_path = '../data/MPII'
	images_path = os.path.join(mpii_path, 'images')
	annot_path = os.path.join(mpii_path, 'annot')

	train_annot = h5py.File(annot_path + '/train.h5', 'r')
	img_names = np.array(train_annot.get('imgname'))
	joints = np.array(train_annot.get('part'))

	test_img = img_names[index]
	test_joints = joints[index]

	test_img_path = os.path.join(images_path, test_img.decode("utf-8"))
	test_im = Image.open(test_img_path)

	connected_pairs = [(0,1), (1, 2), (2,6), (6,3), (3,4), (4,5), 
				   (6,7), (7,8), (8,9), (8,12), (12,11), (11,10), 
				  (8,13), (13, 14), (14, 15)]

	test_im2 = test_im.resize((96,96))
	plt.imshow(test_im2)

	og_rows = np.array(test_im).shape[0]
	og_cols = np.array(test_im).shape[1]

	Ry = (96/og_rows)
	Rx = (96/og_cols)

	for i in range(test_joints.shape[0]):
		plt.scatter(test_joints[i][0]*Rx, test_joints[i][1]*Ry)

	for (p1, p2) in connected_pairs:
		x = [test_joints[p1][0]*Rx, test_joints[p2][0]*Rx]
		y = [test_joints[p1][1]*Ry, test_joints[p2][1]*Ry]
		plt.plot(x, y)
		
	plt.title('i='+ str(i))
	plt.savefig('image_train_with_annot_' + str(index)+'.png')
	plt.close()


def plot_valid_image_with_annotation(index):
	mpii_path = '../data/MPII'
	images_path = os.path.join(mpii_path, 'images')
	annot_path = os.path.join(mpii_path, 'annot')

	train_annot = h5py.File(annot_path + '/valid.h5', 'r')
	img_names = np.array(train_annot.get('imgname'))
	joints = np.array(train_annot.get('part'))

	test_img = img_names[index]
	test_joints = joints[index]

	test_img_path = os.path.join(images_path, test_img.decode("utf-8"))
	test_im = Image.open(test_img_path)

	connected_pairs = [(0,1), (1, 2), (2,6), (6,3), (3,4), (4,5), 
				   (6,7), (7,8), (8,9), (8,12), (12,11), (11,10), 
				  (8,13), (13, 14), (14, 15)]

	test_im2 = test_im.resize((96,96))
	plt.imshow(test_im2)

	og_rows = np.array(test_im).shape[0]
	og_cols = np.array(test_im).shape[1]

	Ry = (96/og_rows)
	Rx = (96/og_cols)

	for i in range(test_joints.shape[0]):
		plt.scatter(test_joints[i][0]*Rx, test_joints[i][1]*Ry)

	for (p1, p2) in connected_pairs:
		x = [test_joints[p1][0]*Rx, test_joints[p2][0]*Rx]
		y = [test_joints[p1][1]*Ry, test_joints[p2][1]*Ry]
		plt.plot(x, y)
		
	plt.title('i='+ str(i))
	plt.savefig('image_valid_with_annot_' + str(index)+'.png')
	plt.close()

def plot_test_image_with_annotation(index):
	mpii_path = '../data/MPII'
	images_path = os.path.join(mpii_path, 'images')
	annot_path = os.path.join(mpii_path, 'annot')

	train_annot = h5py.File(annot_path + '/test.h5', 'r')
	img_names = np.array(train_annot.get('imgname'))
	joints = np.array(train_annot.get('part'))

	test_img = img_names[index]
	test_joints = joints[index]

	test_img_path = os.path.join(images_path, test_img.decode("utf-8"))
	test_im = Image.open(test_img_path)

	connected_pairs = [(0,1), (1, 2), (2,6), (6,3), (3,4), (4,5), 
				   (6,7), (7,8), (8,9), (8,12), (12,11), (11,10), 
				  (8,13), (13, 14), (14, 15)]

	test_im2 = test_im.resize((96,96))
	plt.imshow(test_im2)

	og_rows = np.array(test_im).shape[0]
	og_cols = np.array(test_im).shape[1]

	Ry = (96/og_rows)
	Rx = (96/og_cols)

	for i in range(test_joints.shape[0]):
		plt.scatter(test_joints[i][0]*Rx, test_joints[i][1]*Ry)

	for (p1, p2) in connected_pairs:
		x = [test_joints[p1][0]*Rx, test_joints[p2][0]*Rx]
		y = [test_joints[p1][1]*Ry, test_joints[p2][1]*Ry]
		plt.plot(x, y)
		
	plt.title('i='+ str(i))
	plt.savefig('image_test_with_annot_' + str(index)+'.png')
	plt.close()
from dependencies import *

def load_train_data():
	mpii_path = '../data/MPII'
	images_path = os.path.join(mpii_path, 'images')
	annot_path = os.path.join(mpii_path, 'annot')

	train_annot = h5py.File(annot_path + '/train.h5', 'r')
	train_img_names = np.array(train_annot.get('imgname'))
	train_joints = np.array(train_annot.get('part'))

	train_labels_dict = {}
	train_images_dict = {}

	Ry=0.13333333333333333
	Rx=0.075

	for i in range(len(train_img_names)):
		curr_img = train_img_names[i].decode("utf-8")
		curr_joints = train_joints[i]
		for j in range(len(curr_joints)):
			new_x = Rx * curr_joints[j][0]
			new_y = Ry * curr_joints[j][1]
			curr_joints[j][0] = new_x
			curr_joints[j][1] = new_y
		
		curr_joints = curr_joints.flatten()
		
		train_labels_dict[i] = curr_joints
		train_images_dict[i] = curr_img
	return train_labels_dict, train_images_dict


class MPII_Dataset(data.Dataset):
#       '''Characterizes a dataset for PyTorch'''
	def __init__(self, labels, images, images_path):
		'''Initialization'''
		self.labels = labels
		self.images = images
		
		self.images_path = images_path
		
		self.transform = transforms.Compose(
				[
					transforms.Resize((96, 96)),
					transforms.ToTensor(),
#                     transforms.CenterCrop(10),
				 
				 transforms.Normalize((0.5, 0.5, 0.5), 
									  (0.5, 0.5, 0.5))])

	def __len__(self):
		'''Denotes the total number of samples'''
		return len(self.labels)

	def __getitem__(self, index):
		'''Generates one sample of data'''
		# Select sample
		image_filename = self.images[index]
		path_to_image = os.path.join(self.images_path, image_filename)

		# Load data and get label
		image = Image.open(path_to_image)
		image = self.transform(image).double()
		x = image
		y = torch.tensor(np.array(self.labels[index])).double()

		return x, y



		
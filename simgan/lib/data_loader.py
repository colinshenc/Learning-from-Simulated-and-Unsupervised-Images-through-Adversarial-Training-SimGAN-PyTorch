'''
	DataLoader over rides the torch.utils.data.dataset.Dataset class
	This should be changed dependent on the format of your data sets.
'''

from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import os

class DataLoader(Dataset):
	'''
		DataLoader: class to load the data. Passed into torch.utils.data.dataloader

			Notes*
				1. Input is the path to the directory with all the images
				2. This class is overriding the torch.utils.data.dataset.Dataset class
				3. This class is passed to torch.utils.data.dataloader
	'''
	# used internally
	# load image files, get # of images, load transforms
	def __init__(self, path):
		self.imgs = []


		num_folder_for_training = len(next(os.walk(path))[1])
		print('num of folders {}'.format(num_folder_for_training))


		for person_id in os.listdir(path):
			#print('person_id{}'.format(person_id))
			person_images = os.path.join(path, person_id)

			if os.path.isdir(person_images):
				for img in os.listdir(person_images):
					img = os.path.join(person_images, img)
					#print(img)
					self.imgs.append(img)

			elif os.path.isfile(person_images):
				self.imgs.append(person_images)

			else:
				raise Exception('input neither file nor dir!!!')
		self.data_len = len(self.imgs)

		self.transform = transforms.Compose([transforms.ToTensor()])
		self.path = path	
	
	# used externally
	# prepares the data by oppening image / applying the transform
	def __getitem__(self, index):
		image_file = self.imgs[index]
		#print('img file {}'.format(image_file))
		img_as_img = Image.open(image_file)
		img_as_tensor = self.transform(img_as_img)

		return img_as_tensor, image_file
	
	# used internally / externally
	# get the number of data points
	def __len__(self):
		return self.data_len

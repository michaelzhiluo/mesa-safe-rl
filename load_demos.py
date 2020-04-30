import numpy as np
import pickle
import os.path as osp
import os

def load_demos(demo_dir):
	with open(osp.join("demos", demo_dir, "data.pkl"), "rb") as f:
		data = pickle.load(f)
	return data

if __name__ == '__main__':
	load_demos("reacher")

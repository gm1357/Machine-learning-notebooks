import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from IPython.display import Image
from sklearn.datasets.samples_generator import make_blobs



def check_q1(stuff):
    a = 0
    b = 60
    c = 22.9
    d = 4.53
    e = 511.7

    q1_dict = {
    'number of missing values': a,
    'the mean 5k time in minutes': c,    
    'the mean test score as a raw value': e,
    'number of individuals in the dataset': b
    }
    
    if stuff == q1_dict:
        print("That looks right!")
    
    else:
        print("Oops!  That doesn't look quite right! Try again.")
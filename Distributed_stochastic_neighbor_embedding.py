import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn import manifold

import seaborn as sns
#%matplotlib inline

data = datasets.fetch_openml('mnist_784', version=1, return_X_y=True )

pixel_values, targets = data
targets = targets.astype(int)

single_image = pixel_values[1, :].reshape(28, 28) # take 1 image and reshape it to show the plot

plt.imshow(single_image, cmap = 'gray')

tnse = manifold.TSNE(n_components=2, random_state=42) # 2 Dimensional visualization

transformed_data = tnse.fit_transform(pixel_values[:3000, :]) # transform 3000 images, so eventually we get 3000 images of 2 dim each

tsne_df = pd.DataFrame(np.column_stack((transformed_data, targets[:3000])), columns=["x", "y", "targets"])

tsne_df.loc[:, "targets"] = tsne_df.targets.astype(int)

grid = sns.FacetGrid(tsne_df, hue="targets", size=5)

grid.map(plt.scatter, "x", "y").add_legend()
plt.show()








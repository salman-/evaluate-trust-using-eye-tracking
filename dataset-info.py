import pandas as pd

pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 200)

import numpy as np
import matplotlib.pyplot as plt

categorical_trust_dt = pd.read_csv("Dataset.csv")

print("==================== Number of Labels in each category =======================================")
print(categorical_trust_dt['Label'].value_counts())

fig = plt.figure()
axe = fig.add_axes([0.1, 0.1, 0.8, 0.8])
plt.xticks([1, 2, 3, 4, 5])
axe.hist(categorical_trust_dt['Label'], bins=5)
fig.savefig('./visualization/categorical label histogram.png')
plt.show()

# Labels as 4, or 5 are TRUSTED and 1,2,3 are NOT TRUSTED
categorical_trust_dt['Label'] = categorical_trust_dt['Label'].apply(lambda x: 1 if x > 3 else 0)
binary_trust_dt = categorical_trust_dt
print("==================== Number of Labels in Binary dataset =======================================")
print(binary_trust_dt['Label'].value_counts())

fig = plt.figure()
axe = fig.add_axes([0.1, 0.1, 0.8, 0.8])
plt.xticks([0, 1])
axe.hist(binary_trust_dt['Label'], bins=2)
fig.savefig('./visualization/Binary label histogram.png')
plt.show()

print("==================== Number of features =======================================")
# Exclude participants and labels
print(len(categorical_trust_dt.columns))

print("==================== Dataframe summary ========================================")
print(categorical_trust_dt.describe())

print("==================== Dataframe column types ===================================")
print(categorical_trust_dt.dtypes)

print("==================== Count missing values of each column ======================")
print(categorical_trust_dt.isna().sum())
categorical_trust_dt.dropna(inplace=True)

print("==================== Distribution of each column ==============================")

"""
def get_min_max(data):
    return (np.min(data), np.max(data))


NROWS = 9
NCOLS = 2

fig, axes = plt.subplots(nrows=NROWS, ncols=NCOLS, figsize=(11, 11))

list_of_columns_to_plot = dt.columns[1:-1]
index = 0
for row_index in np.arange(0, NROWS):
    for column_index in np.arange(0, NCOLS):
        axes[row_index, column_index].hist(dt.loc[:, list_of_columns_to_plot[index]],
                                           bins=100,
                                           range=get_min_max(dt.loc[:, list_of_columns_to_plot[index]].values))
        axes[row_index, column_index].set_title(list_of_columns_to_plot[index], size=7)
        axes[row_index, column_index].get_xaxis().set_visible(False)
        index = index + 1

fig.savefig('distributions.png', dpi=200)
plt.tight_layout()
plt.show()
"""
print("==================== line chart of each tuple column (DA,ORIGINAL) ==============================")

DA_ORIGINAL_TUPLES = [
    ('Share of Total Time (%)- DA', 'Share of Total Time (%)- Original', 'Share of Total Time (%)'),
    ('Total duration of fixation in AOI-DA', 'Total duration of fixation in AOI- Original',
     'Total duration of fixation in AOI'),
    ('Average duration of fixation in AOI- DA', 'Average duration of fixation in AOI-Original',
     'Average duration of fixation in AOI'),
    ('Number of fixations in AOI- DA', 'Number of fixations in AOI-Original', 'Number of fixations in AOI'),
    ('Time to first fixation in AOI-DA', 'Time to first fixation in AOI-Original', 'Time to first fixation in AOI'),
    ('Duration of first fixation in AOI-DA', 'Duration of first fixation in AOI-Original',
     'Duration of first fixation in AOI'),
    ('Total duration of Visit-DA', 'Total duration of Visit-Original', 'Total duration of Visit'),
    ('Average duration of Visit-DA', 'Average duration of Visit-Original', 'Average duration of Visit'),
    ('Number of Visits-DA', 'Number of Visits-Original', 'Number of Visits')
]


def draw_histograms(dt, name_to_save_image):
    NROWS = 3
    NCOLS = 3
    fig, axes = plt.subplots(nrows=NROWS, ncols=NCOLS, figsize=(11, 11))
    index = 0
    for row in np.arange(0, NROWS):
        for col in np.arange(0, NCOLS):
            da_column = dt[DA_ORIGINAL_TUPLES[index][0]]
            original_column = dt[DA_ORIGINAL_TUPLES[index][1]]
            axes[row, col].set_title(DA_ORIGINAL_TUPLES[index][2], size=6)

            axes[row, col].hist(original_column, bins=100, label='Original')
            axes[row, col].hist(da_column, bins=100, label='Da')

            axes[row, col].get_xaxis().set_visible(False)
            axes[row, col].get_yaxis().set_visible(False)

            axes[row, col].legend()
            index = index + 1

    plt.tight_layout()
    fig.savefig(name_to_save_image, dpi=200)
    plt.show()


# All categorical data
draw_histograms(categorical_trust_dt, "./visualization/Categorical overlap-distributions.png")

# Only Trusted data (Binary)
draw_histograms(binary_trust_dt[binary_trust_dt['Label'] == 1], "./visualization/TRUSTED overlap-distributions.png")

# Only NOT-Trusted data (Binary)
draw_histograms(binary_trust_dt[binary_trust_dt['Label'] == 0], "./visualization/NOT TRUSTED overlap-distributions.png")

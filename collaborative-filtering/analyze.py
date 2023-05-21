import matplotlib.pyplot as plt
from data import df

# Plot a histogram of the rating values
plt.hist(df["rating"], bins=50)
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.title("Distribution of Movie Ratings")
plt.show()

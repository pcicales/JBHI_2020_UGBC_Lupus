import matplotlib.pyplot as plt

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Control', 'I/II', 'III/IV'
explode = (0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig, axs = plt.subplots(3, 2)
ax = axs[0, 0]
sizes = [0, 80, 20]
ax.pie(sizes, explode=explode, labels=labels, shadow=False, startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

ax = axs[0, 1]
sizes = [0, 80, 20]
ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=False, startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()

print()
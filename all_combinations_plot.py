import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

#IMF Combinations
objects = ('12','13','14','15','16','23','24','25','26','34','35','36','45','46','56',
'123','124','125','126','134','135','136','145','146','156','234','235','236','245','246','256','345','346','356','456',
'1245','1246','1256','1345','1346','1356','1456','2345','2356','2456','3456')
y_pos = np.arange(len(objects))

#Accuracy for each combination
performance = [95.931,90.426,86.492,68.719,78.193,97.374,92.805,92.267,91.956,89.047,86.792,86.907,81.063,76.871,73.337,
    98.24,96.578,96.30,95.762,92.473,89.079,85.174,85.340,81.919,78.790,97.804,97.289,97.478,95.325,94.952,93.182,95.387,93.261,92.944,92.248,
	95.984,96.055,97.027,96.165,95.167,81.414,90.075,95.931,98.140,96.221,97.647]

#Specifying separate colors for each plot
plt.bar(y_pos, performance, align='center', alpha=0.5,color=['black', 'gray','silver','rosybrown','firebrick','red',
'darksalmon','sienna','sandybrown','bisque','tan','gold','darkkhaki',
'olivedrab','chartreuse','palegreen','darkgreen','seagreen',
'mediumspringgreen','lightseagreen','darkcyan','darkturquoise',
'slategray','royalblue','navy','blue','mediumpurple','darkorchid',
'plum','m','palevioletred','crimson','fuchsia','darkviolet',
'reebeccapurple','slateblue','slategrey','cadetblue',
'darkslategray','g','olive','yellowgreen','orange','maroon','lightcoral','goldenrod',
'peru','chocolate','indianred','y','brown','limegreen','lime','teal','tomato','indigo','cornflowerblue'])
plt.xticks(y_pos, objects)
plt.ylabel("Classification Accuracy (%)")
plt.xlabel('IMF Combinations')
#plt.title('Programming language usage')

#plt.show()

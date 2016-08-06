import pandas as pd
import numpy as np

# TODO: Load up the dataset
# Ensuring you set the appropriate header column names
#
servo = pd.read_csv("Datasets\servo.data", names = ('motor', 'screw', 'pgain', 
'vgain', 'class'), header=None)


# TODO: Create a slice that contains all entries
# having a vgain equal to 5. Then print the 
# length of (# of samples in) that slice:
#
servo_vgain = servo[servo.vgain==5]
print (len(servo_vgain))


# TODO: Create a slice that contains all entries
# having a motor equal to E and screw equal
# to E. Then print the length of (# of
# samples in) that slice:
#
# .. your code here ..
servo_motorE_screwE = servo[(servo.motor == 'E') & (servo.screw == 'E')]
print (len(servo_motorE_screwE))

# TODO: Create a slice that contains all entries
# having a pgain equal to 4. Use one of the
# various methods of finding the mean vgain
# value for the samples in that slice. Once
# you've found it, print it:
#
# .. your code here ..
servo_pgain = servo[servo.pgain == 4]
mean_vgain = np.mean(servo_pgain.vgain)
print(mean_vgain)

# TODO: (Bonus) See what happens when you run
# the .dtypes method on your dataframe!




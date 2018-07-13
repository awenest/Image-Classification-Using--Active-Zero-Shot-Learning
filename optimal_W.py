import numpy as np
import os

loc='/media/rishabh/dump_bin/Animals_with_Attributes2/Test'
with open('/media/rishabh/dump_bin/Animals_with_Attributes2/Test/list.txt') as f:
    awa_testclass = f.readlines()
for i in range(len(awa_testclass)):
    awa_testclass[i]=awa_testclass[i][:-1]

print(awa_testclass)
i=0
j=0

for root, subdirs, files in os.walk(loc):
    for filename in files:
        if filename.endswith("npy"):
            if not awa_testclass.__contains__(filename[:-4]):
                filename=root+'/'+filename
                xi = np.load(filename)
                if j == 0:
                    fp = xi
                else:
                    fp = np.vstack((fp, xi))
                j = j + 1

                filename_list=(filename.split('/'))[-1].split('_')[0]
                filename_list=loc+'/'+filename_list+'.npy'
                yi=np.load(filename_list)
                if i==0:
                    tp=yi
                else:
                    tp=np.vstack((tp,yi))
                i=i+1
Xs=fp.T
Ys=tp.T
print(Xs.shape)
print(Ys.shape)
j=0
for labels in awa_testclass:


    x=labels
    x=loc+'/'+x+'.npy'
    a=np.load(x)
    if j==0:
        dp=a
    else:
        dp=np.add(dp,a)
    j=j+1

Y_cap=dp/j
for itr in range(i):
    if itr == 0:
        sp = Y_cap
    else:
        sp = np.vstack((sp, Y_cap))
Ys_cap=sp.T
print(Ys_cap.shape)

lamda1=1000
lamda2=10



first_exp  =  (Xs.dot(Xs.T))
sec_exp   =   lamda1*(np.identity(512))
third_exp_1  =  lamda2*Xs
third_exp_2  =  L.dot(Xs.T)
third_exp =  third_exp_1.dot(third_exp_2)
com_exp = np.add(first_exp,np.add(sec_exp,third_exp))
try:
    inverse = np.linalg.inv(com_exp)
except np.linalg.LinAlgError:
    # Not invertible. Skip this one.
    pass
else:
    # continue with what you were doing
forth_exp=Xs.dot(Ys.T)
fifth_exp=Xs.dot(Ys_cap.T)
last_exp = np.subtract(forth_exp,fifth_exp)

final_exp=inverse.dot(last_exp)



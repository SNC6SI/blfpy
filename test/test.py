import os
import numpy
import blfpy

#myfile = u"d:\\冬季试验数据\\401\\2019-12-26_02-55-04\\401_2019-12-26_10-20-04.blf"
myfile = r"C:\D\02_MatlabRelevant\99_Playground\charge_time\0706\20200608_IC321_500_快充测试009.blf"
myfile = myfile.encode('GBK')

a = blfpy.readFileInfo(myfile)
print(a)
b = blfpy.readFileData(myfile)
print(b)
#c = pyBlfLoad.readFileData(myfile)


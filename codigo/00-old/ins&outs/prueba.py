image= [[0,1,1,0,1,0,0],
	[0,0,0,0,0,0,0],
	[1,0,0,0,0,0,1],
	[1,0,0,0,0,0,0],
	[1,0,0,0,0,0,0],
	[0,0,0,0,0,0,1],
	[0,1,1,0,0,1,1]]


height = len(image)-1  #240
width = len(image[0])-1  #320

InsOuts = []

Sup = []
i = 0


for index,pixel in enumerate(image[0]):	#Fila Superior
	if pixel == 0 and i != 0:
		Sup.append([index-i,index-1])
		i = 0

	elif pixel == 1:
		i = i+1

if pixel == 1:
	InsOuts.append([0,width])


Inf = []
i = 0

for index,pixel in enumerate(image[height]):	#Fila Inferior
	if pixel == 0 and i != 0:
		Inf.append([index-i,index-1])
		i = 0

	elif pixel == 1:
		i = i+1

if pixel == 1:
	InsOuts.append([height,width])


Izq = []
i = 0

for index,pixel in enumerate([row[0] for row in image]):	#Fila Inferior
	if pixel == 0 and i != 0:
		Izq.append([index-i,index-1])
		i = 0

	elif pixel == 1:
		i = i+1

if pixel == 1:
	InsOuts.append([height,0])


Der = []
i = 0

for index,pixel in enumerate([row[width] for row in image]):	#Fila Inferior
	if pixel == 0 and i != 0:
		Der.append([index-i,index-1])
		i = 0

	elif pixel == 1:
		i = i+1

if pixel == 1:
	InsOuts.append([height,width])


for el in Sup:
	InsOuts.append([0,(el[1]-el[0])/2 + el[0]])
for el in Inf:
	InsOuts.append([height,(el[1]-el[0])/2 + el[0]])
for el in Izq:
	InsOuts.append([(el[1]-el[0])/2 + el[0],0])
for el in Der:
	InsOuts.append([(el[1]-el[0])/2 + el[0],width])

print "Sup: "
print Sup
print "Inf: "
print Inf
print "Izq: "
print Izq
print "Der: "
print Der

print "pixeles: "

Result = []
for el in InsOuts:
	if el not in Result:
		Result.append(el)
		

print Result




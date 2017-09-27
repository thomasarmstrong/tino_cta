from DIRAC.Core.Base import Script
Script.parseCommandLine()
from DIRAC.Interfaces.API.Dirac import Dirac


dirac = Dirac()

file_collection = []
for line in open('../vo.cta.in2p3.fr-user-t-tmichael.lfns'):
    line = line.strip()

    if  "prod3b/paranal/classified_events" not in line:
        continue

    if len(file_collection) < 100:
        file_collection.append(line)
    else:
        dirac.removeFile(file_collection, True)
        file_collection = [] 
    
dirac.removeFile(file_collection, True)

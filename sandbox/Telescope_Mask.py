
TelDict = {}

TelDict["LST"]          = range(1,17)    # LST           PE -> ADC about *40
TelDict["MST_NectaCam"] = range(17,29)   # MST NectarCam PE -> ADC about *60
TelDict["MST_FlashCam"] = range(29,53)   # MST FlashCam  PE -> ADC about *55
TelDict["SST_GCT"]      = range(53,66)   # SST GCT       PE -> ADC ?
TelDict["SST_GCT-S"]    = range(66,77)   # SST GCT-S     PE -> ADC ?
TelDict["SST_DC"]       = range(77,102)  # SST DC        PE -> ADC ?
TelDict["SCT"]          = range(102,126) # SCT           PE -> ADC ?


TelDict["MST"]          = range(17,53)   # MST NectarCam and FlashCam  

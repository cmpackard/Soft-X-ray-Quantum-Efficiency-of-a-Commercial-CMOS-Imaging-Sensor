
__all__ = """
    read_dict
    plotIOC
    mda_GetLastFileNum
    mda_CurrentDirectory
    mda_CurrentPrefix
    mda_CurrentRun
    mda_CurrentUser
    ca2flux
    flux2ca
""".split()
    
from epics import caget
from os.path import join
import ast
import numpy as np

def read_dict(FileName,FilePath="/home/beams22/29IDUSER/Documents/User_Macros/Macros_29id/IEX_Dictionaries/"):
    with open(join(FilePath, FileName)) as f:
        for c,line in enumerate(f.readlines()):
            if line[0] == '=':
                lastdate=line[8:16]
            lastline=line
        mydict=ast.literal_eval(lastline)
    return mydict


def plotIOC():
    BranchPV=caget("29id:CurrentBranch.VAL")
    if (BranchPV == 0):
        branch = "ARPES"        # PV = 0 => ARPES
    else:
        branch = "Kappa"        # PV = 1 => RSXS
    return branch  

def mda_GetLastFileNum(scanIOC=None):
    if scanIOC is None:
        scanIOC=plotIOC()
    FileNum  = caget("29id"+scanIOC+":saveData_scanNumber")-1
    return FileNum

def mda_CurrentDirectory(scanIOC=None):
    if scanIOC is None:
        scanIOC=plotIOC()
    Dir=caget("29id"+scanIOC+":saveData_fileSystem",as_string=True)
    subDir=caget("29id"+scanIOC+":saveData_subDir",as_string=True)
    FilePath = Dir +'/'+subDir+"/"
    if FilePath[1]=='/':
        FilePath="/net"+FilePath[1:]
    FilePath=FilePath.replace('//','/') 
    return FilePath   

def mda_CurrentPrefix(scanIOC=None):
    if scanIOC is None:
        scanIOC=plotIOC()
    Prefix=caget("29id"+scanIOC+":saveData_baseName",as_string=True)
    return Prefix
   
def mda_CurrentRun(scanIOC=None):
    if scanIOC is None:
        scanIOC=plotIOC()
    directory = mda_CurrentDirectory(scanIOC)
    m=directory.find('data_29id')+len('data_29id')+2
    current_run=directory[m:m+6]
    return current_run
   
def mda_CurrentUser(scanIOC=None):
    if scanIOC is None:
        scanIOC=plotIOC()
    subdir=caget("29id"+scanIOC+":saveData_subDir",as_string=True)
    m=subdir.find('/mda')
    if m == 0 : current_user='Staff'
    elif m > 0: current_user=subdir[1:m]
    else: current_user="";print("WARNING: MDA_CurrentUser is empty string")
    return current_user 




###############################################################################################
####################################### FLUX CONVERSION #######################################
###############################################################################################




def LoadResponsivityCurve():
    FilePath='/home/beams/29IDUSER/Documents/User_Macros/Macros_29id/IEX_Dictionaries/'
    FileName="DiodeResponsivityCurve"
    data = np.loadtxt(FilePath+FileName, delimiter=' ', skiprows=1)
    return data



def ca2flux(ca,hv=None,p=1):
    curve=LoadResponsivityCurve()
    responsivity=curve[:,0]
    energy=curve[:,1]
    charge = 1.602e-19
    if hv is None:
        hv=caget('29idmono:ENERGY_SP')
        print("\nCalculating flux for:")
        print("   hv = %.1f eV" % hv)
        print("   ca = %.3e Amp" % ca)
    eff=np.interp(hv,energy,responsivity)
    flux = ca/(eff*hv*charge)
    if p is not None:
        print("Flux = %.3e ph/s\n" % flux)
    return flux



def flux2ca(flux,hv=None,p=1):
    curve=LoadResponsivityCurve()
    responsivity=curve[:,0]
    energy=curve[:,1]
    charge = 1.602e-19
    if hv is None:
        hv=caget('29idmono:ENERGY_SP')
        print("\nCalculating current for:")
        print("   hv = %.1f eV" % hv)
        print("   flux = %.3e ph/s" % flux)
    eff=np.interp(hv,energy,responsivity)

    ca = flux*(eff*hv*charge)
    if p is not None:
        print("CA = %.3e Amp/n" % ca)
    return ca

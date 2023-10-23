"""
#ScanFunctions_plot.py
how to copy tabs in stabs and not spaces jupyter

For Fanny:    Converting spaces to tabs from the terminal
cp ScanFunctions_IEX.py ScanFunctions_IEXbackup.py
sed -e 's/    /\t/g' ScanFunctions_IEXbackup.py > ScanFunctions_IEX.py
"""
### Data analysis:
#matplotlib.use('nbagg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from math import *

#from netCDF4 import Dataset
#### Other utilities:
import csv
import datetime
from os import listdir
from os.path import join, isfile, dirname
from scipy.optimize import curve_fit
from scipy.special import erf
import numpy.polynomial.polynomial as poly
import ast


##### APS / 29ID-IEX:
from epics import *
from mda import readMDA,scanDim
from file import *

try:
    from Macros_29id.IEX_nData import * 

except:
    print("IEX_nData not imported")

##########################################

if __name__=='__main__':
    print(__file__)



def today_plot(which='default'):
    dt=datetime.date.today()
    today_year = dt.year
    today_month = dt.month
    today_day = dt.day
    if which == 'slash':
        today_str=("{:02d}/{:02d}/{:04d}".format(today_month,today_day,today_year))
    elif which == 'int':
        today_str=int(("{:04d}{:02d}{:02d}".format(today_year,today_month,today_day)))
    else: # 'default':
        today_str=("{:04d}{:02d}{:02d}".format(today_year,today_month,today_day))
    return today_str 



##############################################################################################################
##############################                  JM Curves and Curve Fitting               ##############################
##############################################################################################################

def gauss(x, *p):
    """
    Function for a guassian where p=[A,x0,sigma]
    f(x)=A*numpy.exp(-(x-x0)**2/(2.*sigma**2)
    fwhm=2.355*sigma
    """
    A, x0, sigma = p
    return A*np.exp(-(x-x0)**2/(2.*sigma**2))

def gauss_p0(y,x):
    """
    Inital guesses for guassian fit
    x and y are np.arrays
    returns p0=[A,x0,sigma]
    """
    A=max(y)
    x0=x[np.where(y == max(y))][0]
    sigma=0.2
    return [A,x0,sigma]

def fit_gauss(x,y,xrange=None):
    """
    x,y=EA_Spectrum(ScanNum, EnergyAxis,FilePath,Prefix)
    x,y,x_name,y_name=mda_1D(ScanNum,DetectorNum)
    
    xrange = None to fit the full range 
                = [x1,x2] to fit a subrange
    returns input_x,fit_y,coeff,var_matrix
    where input_x and fit_y are the sub range x and the fitted data respectivley
    coeff=[A,x0,sigma]
    var_matrix is the fit variance
    """
    if xrange is None:
        input_y=y
        input_x=x
    else:
        index1=np.where(x == x.flat[np.abs(x - xrange[0]).argmin()])[0][0]
        index2=np.where(x == x.flat[np.abs(x - xrange[1]).argmin()])[0][0]
        input_x=x[index1:index2]
        input_y=y[index1:index2]
    coeff, var_matrix = curve_fit(gauss, input_x, input_y, p0=gauss_p0(input_y,input_x))
    fit_y= gauss(input_x, *coeff)
    return input_x,fit_y,coeff,var_matrix


##############################################################################################################
##############################                  Plot Tiff,JPEG,PNG               ##############################
##############################################################################################################


#  filepath='/home/beams/29IDUSER/Documents/User_Folders/Topp/S089.tif'
def plot_image(filepath,h=20,v=10,**kwargs):
    """
    filepath = '/home/beams/29IDUSER/Documents/User_Folders/UserName/TifFile.tif'
    ** kwargs are imshow kwargs:
        cmap = colormap; examples => 'gray','BuPu','Inferno'
        vmin,vmax, adjust max and min of colormap
    """
    
    image = mpimg.imread(filepath)
    plt.figure(figsize=(h,v))
    plt.imshow(image,**kwargs)
    plt.axis('off')
    plt.show()
    


def plot_image2(Filepath1,Filepath2,h=20,v=10):
    """
    filepath = '/home/beams/29IDUSER/Documents/User_Folders/UserName/TifFile.tif'
    """
    print(Filepath1)
    print(Filepath2)
    image1 = mpimg.imread(Filepath1)
    image2 = mpimg.imread(Filepath2)
    plt.figure(figsize=(h,v))
    plt.subplot(1,2,1), plt.imshow(image1,cmap='gray')
    plt.axis('off')
    plt.subplot(1,2,2), plt.imshow(image2,cmap='gray')
    plt.axis('off')
#    fig.subplots_adjust(wspace=0.5,hspace=0.5,left=0.125,right=0.9,top=0.85,bottom=0.15)
    plt.tight_layout()
    plt.show()


##############################################################################################################
##############################                  Extract mda Data               ##############################
##############################################################################################################

## mda_1D(1691,44,1,0,'/net/s29data/export/data_29idb/2018_1/mda_b')???


def mda_unpack(ScanNum,filepath=None,prefix=None,scanIOC=None):
    """ Return data file + dictionary D##:("pv",index##)
    filepath: by default plot scans for the current data folder (as defined in plotIOC() ScanRecord SaveData)
    or specified folder path:
        e.g. filepath='/net/s29data/export/data_29idb/2018_1/mda_b/'
    prefix: by default, uses prefix as defined in ScanRecord
            "mda_" for users, "Kappa_" or "ARPES_" for staff (sometimes)
    """
    if scanIOC is None:
        scanIOC = plotIOC()
    if filepath is None:
        filepath =caget("29id"+scanIOC+":saveData_fileSystem",as_string=True)+caget("29id"+scanIOC+":saveData_subDir",as_string=True)
        if filepath[1]=='/':
            filepath="/net"+filepath[1:]
    if filepath[-1] != '/':
        filepath=filepath+'/'
    if prefix is None:
        prefix=caget("29id"+scanIOC+":saveData_baseName",as_string=True)
    if prefix[-1] != '_':
        prefix=prefix+'_'
    mdaFile=filepath + prefix+'{:04}.mda'.format(ScanNum)
    data_file = readMDA(mdaFile)
    try:
        D={}
        n=len(data_file)-1
        for i in range(0,data_file[n].nd):
            detector=data_file[n].d[i].fieldName
            D[int(detector[1:])]=(data_file[n].d[i].name,i)
        return (data_file,D)
    except:
        pass



def mda_1D(ScanNum,DetectorNum,coeff=1,bckg=0,filepath=None,prefix=None,scanIOC=None):
    """ Return x=positionner and y=detector(DetectorNum)
    for a given detector number DYY (as shown in dview). """
    try:
        (data_file,det)=mda_unpack(ScanNum,filepath,prefix,scanIOC)
        index=det[DetectorNum][1]
        x = data_file[1].p[0].data
        y = data_file[1].d[index].data
        x_name = data_file[1].p[0].name
        y_name = data_file[1].d[index].name
        if type(x_name) == bytes:
             x_name=x_name.decode("utf-8")
        if type(y_name) == bytes:
             y_name=y_name.decode("utf-8")
        n=list(zip(x,y))
        d=[n[i] for i, e in enumerate(n) if e != (0.0,0.0)]
        if len(d)<len(n):
            x=x[:len(d)]
            y=y[:len(d)]
        y = [p if p<1e30 else np.nan for p in y]    
        y=[(y[i]+bckg)*coeff for i, e in enumerate(y)]
        return x,y,x_name,y_name
    except:
        print('error')
        pass


def mda_1D_unscaled(ScanNum,DetectorNum,filepath=None,prefix=None,scanIOC=None):
    """ Return x=positionner and y=detector(DetectorNum)
    for a given detector number DYY (as shown in dview). """
    try:
        (data_file,det)=mda_unpack(ScanNum,filepath,prefix,scanIOC)
        if (data_file,det) == (None,None):
            return(None)
        else:
            index=det[DetectorNum][1]
            x = data_file[1].p[0].data
            y = data_file[1].d[index].data
            x_name = data_file[1].p[0].name
            y_name = data_file[1].d[index].name
            if type(x_name) == bytes:
                 x_name=x_name.decode("utf-8")
            if type(y_name) == bytes:
                 y_name=y_name.decode("utf-8")
            n=list(zip(x,y))
            d=[n[i] for i, e in enumerate(n) if e != (0.0,0.0)]
            if len(d)<len(n):
                x=x[:len(d)]
                y=y[:len(d)]
            bckg=min(y)
            coeff=max(y)-min(y)
            y[y>1e30] = np.nan
            y=[(y[i]-bckg)/coeff for i, e in enumerate(y)]
            return x,y,x_name,y_name
    except:
        pass

def mda_1D_Xindex(ScanNum,DetectorNum,coeff=1,bckg=0,filepath=None,prefix=None,scanIOC=None):
    """ Return x=index and y=detector(DetectorNum)
    for a given detector number DYY (as shown in dview). """
    try:
        (data_file,det)=mda_unpack(ScanNum,filepath,prefix,scanIOC)
        index=det[DetectorNum][1]
        x = data_file[1].d[0].data
        y = data_file[1].d[index].data
        x_name = data_file[1].d[0].name
        y_name = data_file[1].d[index].name
        if type(x_name) == bytes:
             x_name=x_name.decode("utf-8")
        if type(y_name) == bytes:
             y_name=y_name.decode("utf-8")
        n=list(zip(x,y))
        d=[n[i] for i, e in enumerate(n) if e != (0.0,0.0)]
        if len(d)<len(n):
            y=y[:len(d)]
        x=list(range(1,len(y)+1))
        y[y>1e30] = np.nan
        y=[(y[i]*coeff)+bckg for i, e in enumerate(y)]
        return x,y,x_name,y_name
    except:
        pass

def mda_1D_vsDet(ScanNum,DetectorNum,DetectorNum2,coeff=1,bckg=0,filepath=None,prefix=None,scanIOC=None):
    """ Return x=index and y=detector(DetectorNum)
    for a given detector number DYY (as shown in dview). """
    try:
        #print(ScanNum,filepath,prefix,scanIOC)
        (data_file,det)=mda_unpack(ScanNum,filepath,prefix,scanIOC)
        index=det[DetectorNum][1]
        index2=det[DetectorNum2][1]
        x = data_file[1].d[0].data
        x2 = data_file[1].d[index2].data
        y = data_file[1].d[index].data
        x_name = data_file[1].d[0].name
        x2_name = data_file[1].d[index2].name
        y_name = data_file[1].d[index].name
        x = data_file[1].p[0].data
        x2= data_file[1].d[index2].data
        y= data_file[1].d[index].data
        if type(x_name) == bytes:
             x_name=x_name.decode("utf-8")
        if type(y_name) == bytes:
             y_name=y_name.decode("utf-8")
        if type(x2_name) == bytes:
             x2_name=x2_name.decode("utf-8")
        n=list(zip(x,y))
        d=[n[i] for i, e in enumerate(n) if e != (0.0,0.0)]
        if len(d)<len(n):
            y=y[:len(d)]
            x2=x2[:len(d)]
        y[y>1e30] = np.nan
        y=[(y[i]*coeff)+bckg for i, e in enumerate(y)]
        return x2,y,x2_name,y_name
    except:
        pass

def mda_Flux(ScanNum,DetectorNum,EnergyNum,filepath=None,prefix=None,scanIOC=None):
    """ Return x=positionner and y=Flux(DetectorNum)
    for a given diode recorded as detector number DYY (see ## in dview).
    EnergyNum is the detector number for the mono RBV.

    """
    
    try:
        (data_file,det)=mda_unpack(ScanNum,filepath,prefix,scanIOC)
        index=det[DetectorNum][1]
        Eindex=det[EnergyNum][1]
        x = data_file[1].p[0].data
        y = data_file[1].d[index].data
        energy = data_file[1].d[Eindex].data
        x_name = data_file[1].p[0].name
        y_name = data_file[1].d[index].name
        if type(x_name) == bytes:
             x_name=x_name.decode("utf-8")
        if type(y_name) == bytes:
             y_name=y_name.decode("utf-8")
        n=list(zip(x,y))
        d=[n[i] for i, e in enumerate(n) if e != (0.0,0.0)]
        if len(d)<len(n):
            x=x[:len(d)]
            y=y[:len(d)]
        y[y>1e30] = np.nan
        y=[ca2flux(y[i],energy[i],p=None) for (i, e) in enumerate(y)]
        return x,y,x_name,y_name
    except:
        pass



def mda_NormDet(ScanNum,DetectorNum,NormNum,coeff=1,bckg=0,filepath=None,prefix=None,scanIOC=None):
    """ Return x=positionner and y=detector(DetectorNum)
    for a given detector number DYY (as shown in dview). """
    try:
        (data_file,det)=mda_unpack(ScanNum,filepath,prefix,scanIOC)
        index=det[DetectorNum][1]
        index_Norm=det[NormNum][1]
        x = data_file[1].p[0].data
        y= data_file[1].d[index].data
        y_Norm=data_file[1].d[index_Norm].data
        x_name = data_file[1].p[0].name
        y_name = data_file[1].d[index].name#+"_norm:"+str(NormNum)
        if type(x_name) == bytes:
             x_name=x_name.decode("utf-8")
        if type(y_name) == bytes:
             y_name=y_name.decode("utf-8")

        n=list(zip(x,y))
        d=[n[i] for i, e in enumerate(n) if e != (0.0,0.0)]
        if len(d)<len(n):
            x=x[:len(d)]
            y=y[:len(d)]
        y[y>1e30] = np.nan
        y=[y[i]/y_Norm[i] for i, e in enumerate(y)]
        return x,y,x_name,y_name
    except:
        pass

def mda_DivScan(ScanNum1,DetectorNum1,ScanNum2,DetectorNum2,coeff=1,bckg=0,filepath=None,prefix=None,scanIOC=None):
    """ Return x=positionner and y=detector(DetectorNum)
    for a given detector number DYY (as shown in dview). """
    try:
        (data_file1,det1)=mda_unpack(ScanNum1,filepath,prefix,scanIOC)
        index1=det1[DetectorNum1][1]
        (data_file2,det2)=mda_unpack(ScanNum2,filepath,prefix,scanIOC)
        index2=det2[DetectorNum2][1]
        x1 = data_file1[1].p[0].data
        y1= data_file1[1].d[index1].data
        y2= data_file2[1].d[index2].data
        x_name = data_file1[1].p[0].name
        y_name = data_file1[1].d[index1].name+"_norm:"+str(ScanNum2)
        if type(x_name) == bytes:
             x_name=x_name.decode("utf-8")
        if type(y_name) == bytes:
             y_name=y_name.decode("utf-8")

        n=list(zip(x1,y1))
        d=[n[i] for i, e in enumerate(n) if e != (0.0,0.0)]
        if len(d)<len(n):
            x1=x1[:len(d)]
            y1=y1[:len(d)]
        y[y>1e30] = np.nan
        y=[y1[i]/y2[i] for i, e in enumerate(y1)]
        return x1,y,x_name,y_name
    except:
        pass



def mda_2D(ScanNum,DetectorNum,filepath=None,prefix=None,scanIOC=None):
    """ Return x=positionner and y=detector(DetectorNum)
    for a given detector number DYY (as shown in dview). """
    try:
        (data_file,det)=mda_unpack(ScanNum,filepath,prefix,scanIOC)
        index=det[DetectorNum][1]
        x_temp = data_file[2].p[0].data
        y_temp = data_file[1].p[0].data
        z_temp = data_file[2].d[index].data
        x_name = data_file[2].p[0].name
        y_name = data_file[1].p[0].name
        z_name = data_file[2].d[index].name
        if type(x_name) == bytes:
             x_name=x_name.decode("utf-8")
        if type(y_name) == bytes:
             y_name=y_name.decode("utf-8")
        if type(z_name) == bytes:
             z_name=z_name.decode("utf-8")

        n=list(zip(x_temp,y_temp,z_temp))
        d=[n[i] for i, e in enumerate(n) if e != (0.0,0.0,0.0)]
        if len(d)<len(n):
            x_temp=x_temp[:len(d)]
            y_temp=y_temp[:len(d)]
            z_temp=z_temp[:len(d)]
        x = x_temp[0]
        y = y_temp
        z[z>1e30] = np.nan
        z = np.asarray(z_temp)     #2-D array
        return x,y,z,x_name,y_name,z_name
    except:
        pass


###############################################################################################
####################################         PLOT MDA        ###################################
###############################################################################################




def plot_mda2D(ScanNum,DetectorNum,title=None,color=None,filepath=None,prefix=None):
    try:
        x,y,z,xName,yName,zName=mda_2D(ScanNum,DetectorNum,filepath,prefix)
        fig, ax0 = plt.subplots()
        if color is None:
            color='gnuplot'
        img = ax0.imshow(z, cmap=color, interpolation = 'nearest', extent = [min(x), max(x), max(y), min(y)], aspect = 'auto')
        fig.colorbar(img)
        if title is None:
            plt.title(zName)
        else:
            plt.title(title)
        ax0.set_xlabel(xName)
        ax0.set_ylabel(yName)
    #    ax0.set_ylim(y.max(),y.min())
        plt.show()
    except:
        pass



def plot_mda_series(*ScanDet,**kwArg):
    """plot_mda_series(1217, 1226,   1,   39 ,0.025, **kwArg)
                    (first,last,countby,det,offset,**kwArg)
    Optional data analysis keywords arguments:
        Flux conversion (for diode only): flux= 3(User) or 25(Staff).
        Norm option: norm = 'yes' normalizes all the scans to 1 (default: None)
        NormDet= 1 for SR current, 14  for Mesh (D-branch); Normalize by the mesh does not work with norm='yes'
    Optional graphical keywords arguments:
        sizeH = 1,1.5,... increase horizontal figure size
        sizeV = 1,1.5,... increase vertical figure size
        marker = 'x','+','o','v','^','D',...    (default:None)
        markersize = 1,2,3,...        (default: 5)
        linewidth = 0.5,1,2...         (default: 1)
        linestyle = '-','--','-.',':'     (default: solid '-')
        color = 'r','b','m','c','g','y'...    (default: jupyter colors series)
        legend = 'best',Supper left', 'lower right'...        (default: None)
        log = 'log'   (default: None = linear)
        xrange = [x1,x2]   (default: None = Autoscale)
        yrange = [y1,y2]   (default: None = Autoscale)
        xlabel = 'mxLabel'        (default: pv name)
        ylabel = 'myLabel'        (default: pv name)
        ytickstyle = 'sci' for y axes    (default: 'plain')
        xtickstyle = 'sci' for y axes    (default: 'plain')
    filepath: by default plot scans for the current data folder (as defined in plotIOC() ScanRecord SaveData)
    or specified folder path:
        e.g. user : filepath='/net/s29data/export/data_29idc/2018_2/UserName/mda/'
        e.g. staff: filepath='/net/s29data/export/data_29idb/2018_2/mda/'
    prefix: by default, uses prefix as defined in ScanRecord ("mda_")
    scanIOC: by default, uses the IOC for the current branch as define in plotIOC()
    """

    if type(ScanDet[0]) is not tuple:
        ScanDet=(tuple(ScanDet),)
        m=1
    else: m= len(ScanDet)

    scanlist=""
    j=0
    offset=0
    for n in range(m):
        print(n)
        print(m)
        print(ScanDet)
        det=ScanDet[n][3]
        if len(ScanDet)>4 and isinstance(ScanDet[n][3],str)== False:
            offset=ScanDet[n][4]
        for scanNum in range(ScanDet[n][0],ScanDet[n][1]+ScanDet[n][2],ScanDet[n][2]):
            scanlist+=str(scanNum)+',(det,1,'+str(offset)+'),'
            j+=offset
        cmd="plot_mda("+scanlist
    if kwArg is not None:
        for key, value in list(kwArg.items()):
            if type(value) == str:
                cmd=cmd+(key+'=\"'+value+'\",')
            else:
                cmd=cmd+(key+'='+str(value)+',')
    if cmd[-1]==",":
        cmd=cmd[:-1]
    cmd=cmd+")"
    if kwArg is not None:
        for key, value in list(kwArg.items()):
            if key=='q':
                print('det=',det)
                print(cmd)
    exec(cmd)



def plot_mda(*ScanDet,**kwArg):

    """
    Plot mda scans: *ScanDet = (scan,det,scan,det...),(scan,det,scan,det...),title=(subplot_title1,subplot_title2)
                             =            subplot1,                subplot2
    Optional data analysis keywords arguments:
        Flux conversion (for diode only): flux= 3(D## for mono rbv, typically 3).
        Norm option: norm = 'yes' normalizes all the scans to 1 (default: None)
        NormDet = 1 for SR current, 14  for Mesh (D-branch); Normalize by the mesh does not work with norm='yes'
        DivScan = ?
    Optional graphical keywords arguments:
        sizeH = 1,1.5,... increase horizontal figure size
        sizeV = 1,1.5,... increase vertical figure size
        marker = 'x','+','o','v','^','D',...    (default:None)
        markersize = 1,2,3,...        (default: 5)
        linewidth = 0.5,1,2...         (default: 1)
        linestyle = '-','--','-.',':'     (default: solid '-')
        color = 'r','b','m','c','g','y'...    (default: jupyter colors series)
        legend = 'best',Supper left', 'lower right'...        (default: None)
        log = 'log'   (default: None = linear)
        xrange = [x1,x2]   (default: None = Autoscale)
        yrange = [y1,y2]   (default: None = Autoscale)
        xlabel = 'mxLabel'        (default: pv name)
        ylabel = 'myLabel'        (default: pv name)
        ytickstyle = 'sci' for y axes    (default: 'plain')
        xtickstyle = 'sci' for y axes    (default: 'plain')
    filepath: by default plot scans for the current data folder (as defined in plotIOC() ScanRecord SaveData)
    or specified folder path:
        e.g. user : filepath='/net/s29data/export/data_29idc/2018_2/UserName/mda/'
        e.g. staff: filepath='/net/s29data/export/data_29idb/2018_2/mda/'
    prefix: by default, uses prefix as defined in ScanRecord ("mda_")
    scanIOC: by default, uses the IOC for the current branch as define in plotIOC()
    """

    args={
        'marker':None,
        'markersize':5,
        'linewidth':1,
        'linestyle':'-',
        'color':None,
        'nticks':None,
        'sizeH':1,
        'sizeV':1,
        'title':'',
        'filepath':"C:\\Users\\Colin Packard\\PythonWorkspace\\Argonne12_01\\MDA_TEST/",
        'prefix':"Kappa_",
        'norm':None,
        'flux':None,
        'NormDet':None,
        'scanIOC':None,
        'legend':None,
        'vs_index':None,
        'vs_det':None,
        'xrange':[None,None],
        'yrange':[None,None],
        'save':True
    }
    
    args.update(kwArg)
    
    mkr=args['marker']
    ms=args['markersize']
    lw=args['linewidth']
    ls=args['linestyle']
    c=args['color']
    path=args['filepath']
    prefix=args['prefix']
    scanIOC=args['scanIOC']
    save=args['save']
    
    if 'legend' in args:
        if args['legend'] == 'center left':
            hsize=7
            
    if type(ScanDet[0]) is not tuple:
        ScanDet=(tuple(ScanDet),)
        m=1
    else: m= len(ScanDet)

    def SubplotsLayout(m):
        if m >1:
            ncols=2
        else:
            ncols=1
        nrows=max(sum(divmod(m,2)),1)
        hsize=ncols*5*args['sizeH']
        vsize=nrows*4*args['sizeV']
        if nrows==1: vsize=nrows*3.5*args['sizeV']
        return nrows,ncols,hsize,vsize

    try:
        nrows,ncols,hsize,vsize=SubplotsLayout(m)

        fig, axes = plt.subplots(nrows,ncols,figsize=(hsize,vsize))    # HxV
        axes=np.array(axes)

        for (n,ax) in zip(list(range(m)),axes.flat):
            for (i,j) in zip(ScanDet[n][0::2],ScanDet[n][1::2]):
                if type(j) is tuple:
                    p,k,l=j
                    x,y,x_name,y_name=mda_1D(i,p,k,l,path,prefix)
                elif args['flux'] is not None:
                    x,y,x_name,y_name=mda_Flux(i,j,args['flux'],path,prefix,scanIOC)
                elif args['norm'] is not None:
                    x,y,x_name,y_name=mda_1D_unscaled(i,j,path,prefix,scanIOC)
                elif args['NormDet'] is not None:
                    x,y,x_name,y_name=mda_NormDet(i,j,args['NormDet'],1,0,path,prefix,scanIOC)
                elif args['vs_index'] is not None:
                    x,y,x_name,y_name=mda_1D_Xindex(i,j,1,0,path,prefix)
                elif args['vs_det'] is not None:
                    x,y,x_name,y_name=mda_1D_vsDet(i,j,args['vs_det'],1,0,path,prefix)
                #elif DivScan is not None:
                #    x,y,x_name,y_name=mda_DivScan(i,j,DivScan,DivDet,1,0,path,prefix,scanIOC)
                else:
                    x,y,x_name,y_name=mda_1D(i,j,1,0,path,prefix,scanIOC)
                ax.plot(x,y,label="mda_"+str(i),color=c,marker=mkr,markersize=ms,linewidth=lw,linestyle=ls)
                ax.grid(color='lightgray', linestyle='-', linewidth=0.5)
                
                #modifying graph
                if args['legend'] != None:
                    if args['legend'] == 'center left':
                        box = ax.get_position()
                        ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
                        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                        myleft=0.1
                        myright=0.55
                    else:
                        ax.legend(args['legend'], frameon=True)
                if 'ytickstyle' in args:
                    ax.ticklabel_format(style=args['ytickstyle'], axis='y', scilimits=(0,0))
                if 'xtickstyle' in args:
                    ax.ticklabel_format(style=args['xtickstyle'], axis='x', scilimits=(0,0))
                if 'log' in args:
                    ax.set_yscale('log')
                if args['xrange'] != [None,None]:
                    ax.set_xlim(args['xrange'][0],args['xrange'][1])
                if args['yrange'] != [None,None]:
                    ax.set_ylim(args['yrange'][0],args['yrange'][1])
                if 'xlabel' in args:
                    x_name=args['xlabel']
                if 'ylabel' in args:
                    y_name=args['ylabel']
                if args['nticks'] != None: 
                    tck=args['nticks']-1
                    ax.locator_params(tight=None,nbins=tck)


            if 'title' in args:
                mytitle=args['title']
                if type(mytitle) is not tuple:
                    ax.set(xlabel=x_name,ylabel=y_name,title=mytitle)
                else:
                    p=len(mytitle)
                    if n == p:
                        ax.set(xlabel=x_name,ylabel=y_name,title='')
                    else:
                        ax.set(xlabel=x_name,ylabel=y_name,title=mytitle[n])
        #        ax.text(0.5, 1.025,mytitle,horizontalalignment='center',fontsize=14,transform = ax.transAxes)
            ax.grid(color='lightgray', linestyle='-', linewidth=0.5)
                
        if ncols==1 and nrows==1 and kwArg.get('legend')!='center left':
            fig.subplots_adjust(wspace=0.5,hspace=0.5,left=0.25,right=0.88,top=0.85,bottom=0.22)
        elif ncols==1 and kwArg.get('legend')=='center left':
            fig.subplots_adjust(wspace=0.5,hspace=0.5,left=myleft,right=myright,top=0.85,bottom=0.22)
        else:
            fig.subplots_adjust(wspace=0.5,hspace=0.5,left=0.125,right=0.9,top=0.85,bottom=0.15)

        plt.tight_layout()
        if save:
            try:
                fname=join('/home/beams/29IDUSER/Documents/User_Folders/','lastfigure.png')
                print(fname)
                plt.savefig(fname)
            except:
                pass
        plt.show()
    except:
        pass


def plot_mda_deconstruct(*ScanDet,**kwArg):

    """
    Plot mda scans: *ScanDet = (scan,det,scan,det...),(scan,det,scan,det...),title=(subplot_title1,subplot_title2)
                             =            subplot1,                subplot2
    Optional data analysis keywords arguments:
        Flux conversion (for diode only): flux= 3(D## for mono rbv, typically 3).
        Norm option: norm = 'yes' normalizes all the scans to 1 (default: None)
        NormDet = 1 for SR current, 14  for Mesh (D-branch); Normalize by the mesh does not work with norm='yes'
        DivScan = ?
    Optional graphical keywords arguments:
        sizeH = 1,1.5,... increase horizontal figure size
        sizeV = 1,1.5,... increase vertical figure size
        marker = 'x','+','o','v','^','D',...    (default:None)
        markersize = 1,2,3,...        (default: 5)
        linewidth = 0.5,1,2...         (default: 1)
        linestyle = '-','--','-.',':'     (default: solid '-')
        color = 'r','b','m','c','g','y'...    (default: jupyter colors series)
        legend = 'best',Supper left', 'lower right'...        (default: None)
        log = 'log'   (default: None = linear)
        xrange = [x1,x2]   (default: None = Autoscale)
        yrange = [y1,y2]   (default: None = Autoscale)
        xlabel = 'mxLabel'        (default: pv name)
        ylabel = 'myLabel'        (default: pv name)
        ytickstyle = 'sci' for y axes    (default: 'plain')
        xtickstyle = 'sci' for y axes    (default: 'plain')
    filepath: by default plot scans for the current data folder (as defined in plotIOC() ScanRecord SaveData)
    or specified folder path:
        e.g. user : filepath='/net/s29data/export/data_29idc/2018_2/UserName/mda/'
        e.g. staff: filepath='/net/s29data/export/data_29idb/2018_2/mda/'
    prefix: by default, uses prefix as defined in ScanRecord ("mda_")
    scanIOC: by default, uses the IOC for the current branch as define in plotIOC()
    """

    args={
        'marker':None,
        'markersize':5,
        'linewidth':1,
        'linestyle':'-',
        'color':None,
        'nticks':None,
        'sizeH':1,
        'sizeV':1,
        'title':'',
        'filepath':"C:\\Users\\Colin Packard\\PythonWorkspace\\Argonne12_01\\MDA_TEST/",
        'prefix':"Kappa_",
        'norm':None,
        'flux':None,
        'NormDet':None,
        'scanIOC':None,
        'legend':None,
        'vs_index':None,
        'vs_det':None,
        'xrange':[None,None],
        'yrange':[None,None],
        'save':True
    }
    
    args.update(kwArg)
    
    mkr=args['marker']
    ms=args['markersize']
    lw=args['linewidth']
    ls=args['linestyle']
    c=args['color']
    path=args['filepath']
    prefix=args['prefix']
    scanIOC=args['scanIOC']
    save=args['save']
    
    if 'legend' in args:
        if args['legend'] == 'center left':
            hsize=7
            
    if type(ScanDet[0]) is not tuple:
        ScanDet=(tuple(ScanDet),)
        m=1
    else: m= len(ScanDet)

    def SubplotsLayout(m):
        if m >1:
            ncols=2
        else:
            ncols=1
        nrows=max(sum(divmod(m,2)),1)
        hsize=ncols*5*args['sizeH']
        vsize=nrows*4*args['sizeV']
        if nrows==1: vsize=nrows*3.5*args['sizeV']
        return nrows,ncols,hsize,vsize

    try:
        nrows,ncols,hsize,vsize=SubplotsLayout(m)

        fig, axes = plt.subplots(nrows,ncols,figsize=(hsize,vsize))    # HxV
        axes=np.array(axes)

        for (n,ax) in zip(list(range(m)),axes.flat):
            for (i,j) in zip(ScanDet[n][0::2],ScanDet[n][1::2]):
                if type(j) is tuple:
                    p,k,l=j
                    x,y,x_name,y_name=mda_1D(i,p,k,l,path,prefix)
                    print("1")
                elif args['flux'] is not None:
                    x,y,x_name,y_name=mda_Flux(i,j,args['flux'],path,prefix,scanIOC)
                    print("2")
                elif args['norm'] is not None:
                    x,y,x_name,y_name=mda_1D_unscaled(i,j,path,prefix,scanIOC)
                    print("3")
                elif args['NormDet'] is not None:
                    x,y,x_name,y_name=mda_NormDet(i,j,args['NormDet'],1,0,path,prefix,scanIOC)
                    print("4")
                elif args['vs_index'] is not None:
                    x,y,x_name,y_name=mda_1D_Xindex(i,j,1,0,path,prefix)
                    print("5")
                elif args['vs_det'] is not None:
                    x,y,x_name,y_name=mda_1D_vsDet(i,j,args['vs_det'],1,0,path,prefix)
                    print("6")
                #elif DivScan is not None:
                #    x,y,x_name,y_name=mda_DivScan(i,j,DivScan,DivDet,1,0,path,prefix,scanIOC)
                else:
                    x,y,x_name,y_name=mda_1D(i,j,1,0,path,prefix,scanIOC)
                    print("7")
                    print("i: "+str(i))
                    print("j: " + str(j))
                    print(scanIOC)
                ax.plot(x,y,label="mda_"+str(i),color=c,marker=mkr,markersize=ms,linewidth=lw,linestyle=ls)
                ax.grid(color='lightgray', linestyle='-', linewidth=0.5)
                
                #modifying graph
                if args['legend'] != None:
                    if args['legend'] == 'center left':
                        box = ax.get_position()
                        ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
                        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                        myleft=0.1
                        myright=0.55
                    else:
                        ax.legend(args['legend'], frameon=True)
                if 'ytickstyle' in args:
                    ax.ticklabel_format(style=args['ytickstyle'], axis='y', scilimits=(0,0))
                if 'xtickstyle' in args:
                    ax.ticklabel_format(style=args['xtickstyle'], axis='x', scilimits=(0,0))
                if 'log' in args:
                    ax.set_yscale('log')
                if args['xrange'] != [None,None]:
                    ax.set_xlim(args['xrange'][0],args['xrange'][1])
                if args['yrange'] != [None,None]:
                    ax.set_ylim(args['yrange'][0],args['yrange'][1])
                if 'xlabel' in args:
                    x_name=args['xlabel']
                if 'ylabel' in args:
                    y_name=args['ylabel']
                if args['nticks'] != None: 
                    tck=args['nticks']-1
                    ax.locator_params(tight=None,nbins=tck)


            if 'title' in args:
                mytitle=args['title']
                if type(mytitle) is not tuple:
                    ax.set(xlabel=x_name,ylabel=y_name,title=mytitle)
                else:
                    p=len(mytitle)
                    if n == p:
                        ax.set(xlabel=x_name,ylabel=y_name,title='')
                    else:
                        ax.set(xlabel=x_name,ylabel=y_name,title=mytitle[n])
        #        ax.text(0.5, 1.025,mytitle,horizontalalignment='center',fontsize=14,transform = ax.transAxes)
            ax.grid(color='lightgray', linestyle='-', linewidth=0.5)
                
        if ncols==1 and nrows==1 and kwArg.get('legend')!='center left':
            fig.subplots_adjust(wspace=0.5,hspace=0.5,left=0.25,right=0.88,top=0.85,bottom=0.22)
        elif ncols==1 and kwArg.get('legend')=='center left':
            fig.subplots_adjust(wspace=0.5,hspace=0.5,left=myleft,right=myright,top=0.85,bottom=0.22)
        else:
            fig.subplots_adjust(wspace=0.5,hspace=0.5,left=0.125,right=0.9,top=0.85,bottom=0.15)

        plt.tight_layout()
        if save:
            try:
                fname=join('/home/beams/29IDUSER/Documents/User_Folders/','lastfigure.png')
                print(fname)
                plt.savefig(fname)
            except:
                pass
        plt.show()
    except:
        pass

def plot_mda_lists(*ScanDet,**kwArg):
    """
    Plot mda scans: *ScanDet = (scanNum_list,detNum_list),(scanNum_list,detNum_list)
                             =            subplot1,                subplot2
    Optional data analysis keywords arguments:
        Flux conversion (for diode only): flux= 3(User) or 25(Staff).
        Norm option: norm = 'yes' normalizes all the scans to 1 (default: None)
        NormDet= 1 for SR current, 14  for Mesh (D-branch); Normalize by the mesh does not work with norm='yes'
    Optional graphical keywords argudef plot_mdaments:
        sizeH = 1,1.5,... increase horizontal figure size
        sizeV = 1,1.5,... increase vertical figure size
        marker = 'x','+','o','v','^','D',...    (default:None)
        markersize = 1,2,3,...        (default: 5)
        linewidth = 0.5,1,2...         (default: 1)
        linestyle = '-','--','-.',':'     (default: solid '-')
        color = 'r','b','m','c','g','y'...    (default: jupyter colors F)
        legend = 'best',Supper left', 'lower right'...        (default: None)
        log = 'log'   (default: None = linear)
        xrange = [x1,x2]   (default: None = Autoscale)
        yrange = [y1,y2]   (default: None = Autoscale)
        xlabel = 'mxLabel'        (default: pv name)
        ylabel = 'myLabel'        (default: pv name)
        ytickstyle = 'sci' for y axes    (default: 'plain')
        xtickstyle = 'sci' for y axes    (default: 'plain')
    filepath: by default plot scans for the current data folder (as defined in plotIOC() ScanRecord SaveData)
    or specified folder path:
        e.g. user : filepath='/net/s29data/export/data_29idc/2018_2/UserName/mda/'
        e.g. staff: filepath='/net/s29data/export/data_29idb/2018_2/mda/'
    prefix: by default, uses prefix as defined in ScanRecord ("mda_")
    scanIOC: by default, uses the IOC for the current branch as define in plotIOC()
    """
    
    args={
        'marker':None,
        'markersize':5,
        'linewidth':1,
        'linestyle':'-',
        'color':None,
        'nticks':None,
        'sizeH':1,
        'sizeV':1,
        'title':'',
        'filepath':"C:\\Users\\Colin Packard\\PythonWorkspace\\Argonne12_01\\MDA_TEST\\",
        'prefix':"Kappa_",
        'norm':None,
        'flux':None,
        'NormDet':None,
        'scanIOC':None,
        'legend':None,
        'vs_index':None,
        'xrange':[None,None],
        'yrange':[None,None]
    }
    
    args.update(kwArg)
    
    mkr=args['marker']
    ms=args['markersize']
    lw=args['linewidth']
    ls=args['linestyle']
    c=args['color']
    path=args['filepath']
    prefix=args['prefix']
    scanIOC=args['scanIOC']


    
    if 'legend' in args:
        if args['legend'] == 'center left':
            hsize=7
            
    #setting up the subplot
    if type(ScanDet[0]) is not tuple:
        ScanDet=(tuple(ScanDet),)
        m=1
    else: m= len(ScanDet)
        
    def SubplotsLayout(m):
        if m >1:
            ncols=2
        else:
            ncols=1
        nrows=max(sum(divmod(m,2)),1)
        hsize=ncols*5*args['sizeH']
        vsize=nrows*4*args['sizeV']
        if nrows==1: vsize=nrows*3.5*args['sizeV']
        return nrows,ncols,hsize,vsize
    
    try:
        nrows,ncols,hsize,vsize=SubplotsLayout(m)
        fig, axes = plt.subplots(nrows,ncols,figsize=(hsize,vsize))    # HxV
        axes=np.array(axes)


        for (n,ax) in zip(list(range(m)),axes.flat): #n=subplot tuple
            scanNum_list=ScanDet[n][0]
            detNum_list=ScanDet[n][1]

            if type(scanNum_list) is int:
                scanNum_list=[scanNum_list]
            if type(detNum_list) is int:
                detNum_list=[detNum_list]
                for i in range(1,len(scanNum_list)):
                    detNum_list.append(detNum_list[0])
            if type(args['filepath']) is not list:
                filepath_list=[args['filepath']]
                for i in range(1,len(scanNum_list)):
                    filepath_list.append(filepath_list[0])
            else: filepath_list=args['filepath']
            if type(args['prefix']) is not list:
                prefix_list=[args['prefix']]
                for i in range(1,len(scanNum_list)):
                    prefix_list.append(prefix_list[0]) 
            else: prefix_list=args['prefix']
            if type(args['scanIOC']) is not list:
                scanIOC_list=[args['scanIOC']]
                for i in range(1,len(scanNum_list)):
                    scanIOC_list.append(scanIOC_list[0]) 
            else: scanIOC_list=args['scanIOC']
            #loading the data
            for index in range(0,len(scanNum_list)):
                i=scanNum_list[index]
                j=detNum_list[index]
                path=filepath_list[index]
                prefix=prefix_list[index]
                scanIOC=scanIOC_list[index]
                #print(i)
                if type(j) is tuple:
                    p,k,l=j
                    x,y,x_name,y_name=mda_1D(i,p,k,l,path,prefix)
                elif args['flux'] is not None:
                    x,y,x_name,y_name=mda_Flux(i,j,args['flux'],path,prefix,scanIOC)
                elif args['norm'] is not None:
                    x,y,x_name,y_name=mda_1D_unscaled(i,j,path,prefix,scanIOC)
                elif args['NormDet'] is not None:
                    x,y,x_name,y_name=mda_NormDet(i,j, args['NormDet'],1,0,path,prefix,scanIOC)
                else:
                    x,y,x_name,y_name=mda_1D(i,j,1,0,path,prefix,scanIOC)
                #adding to graph
                ax.grid(color='lightgray', linestyle='-', linewidth=0.5)
                ax.plot(x,y,label="mda_"+str(i),color=c,marker=mkr,markersize=ms,linewidth=lw,linestyle=ls)

            #modifying graph
            if args['legend'] != None:
                if args['legend'] == 'center left':
                    box = ax.get_position()
                    ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
                    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    myleft=0.1
                    myright=0.55
                else:
                    ax.legend(args['legend'], frameon=True)
            if 'ytickstyle' in args:
                ax.ticklabel_format(style=args['ytickstyle'], axis='y', scilimits=(0,0))
            if 'xtickstyle' in args:
                ax.ticklabel_format(style=args['xtickstyle'], axis='x', scilimits=(0,0))
            if 'log' in args:
                ax.set_yscale('log')
            if args['xrange'] != [None,None]:
                ax.set_xlim(args['xrange'][0],args['xrange'][1])
            if args['yrange'] != [None,None]:
                ax.set_ylim(args['yrange'][0],args['yrange'][1])
            if 'xlabel' in args:
                x_name=args['xlabel']
            if 'ylabel' in args:
                y_name=args['ylabel']
            if args['nticks'] != None: 
                tck=args['nticks']-1
                ax.locator_params(tight=None,nbins=tck)

            if 'title' in args:
                if type(args['title']) is not str:
                    mytitle=args['title'][n]
                else:
                    mytitle=args['title']
            ax.set(xlabel=x_name,ylabel=y_name,title=mytitle)
        #adjusting subplots
        if ncols==1 and nrows==1 and kwArg.get('legend')!='center left':
            fig.subplots_adjust(wspace=0.5,hspace=0.5,left=0.25,right=0.88,top=0.85,bottom=0.22)
        elif ncols==1 and kwArg.get('legend')=='center left':
            fig.subplots_adjust(wspace=0.5,hspace=0.5,left=myleft,right=myright,top=0.85,bottom=0.22)
        else:
            fig.subplots_adjust(wspace=0.5,hspace=0.5,left=0.125,right=0.9,top=0.85,bottom=0.15)
        #show plot
        plt.tight_layout()
        plt.show()

    except:
        pass












###############################################################################################
##################################### FR curves fitting #######################################
###############################################################################################


def fit_mda(scannum,det,FWHM_or_PolyOrder,fct,hkl_positionner=False,xrange=None,title='',marker='x',graph=True, filepath=None,prefix=None,scanIOC=None):
    """
    fct= 'gauss','lorz','erf','poly','box'
    hkl_positionner = 'H','K','L','tth','th','chi','phi'
    """

    if hkl_positionner == False:
        x,y,x_name,y_name=mda_1D(scannum,det,1,0,filepath,prefix,scanIOC)
    if hkl_positionner:
        d={'h':46,'k':47,'l':48,'tth':54,'th':55,'chi':56,'phi':57}
        x,y,x_name,y_name=mda_1D_vsDet(scannum,det,d[hkl_positionner.lower()],1,0,filepath,prefix,scanIOC)

    if scanIOC is None:
        scanIOC = plotIOC()

    if scanIOC == 'Kappa' and fct != 'poly':
        try:
            title=title + ' centroid = '+str(centroid_avg(scannum))
        except:
            pass

    def closest(lst, K):
        return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))] 

    def gaussian(x,*p):
        amp, cen, wid, bkgd = p
        return bkgd+amp*np.exp(-np.power(x - cen, 2.) / (2 * np.power(wid, 2.)))

    def lorentzian( x, *p):
        amp, x0, gam, bkgd =p
        return bkgd+amp * gam**2 / ( gam**2 + ( x - x0 )**2)

    def step(z,amp=1,bkgd=0,z0=0,width=1):
        return (amp*erf((z-z0)/width))+bkgd

    def box(x, *p):
        height, center, width ,bkgd = p
        return bkgd+height*(center-width/2 < x)*(x < center+width/2)



    if xrange is not None:
        x1=closest(x,xrange[0])
        x2=closest(x,xrange[1])
        xrange=[x.index(x1),x.index(x2)]
        xmin=min(xrange)
        xmax=max(xrange)
        xcrop=x[xmin:xmax]
        ycrop=y[xmin:xmax]
        xdata = np.array(xcrop)
        ydata = np.array(ycrop)
    else:
        xdata = np.array(x)
        ydata = np.array(y)


    Imax=np.max(ydata)
    #print(xdata)
    #print(ydata)
    x0=xdata[np.where(ydata==Imax)][0]
    A0=xdata[0]
    
    nPt_fit=200
    xfit =  np.linspace(min(x),max(x), nPt_fit)
    

    if fct == 'gauss':
        FWHM=FWHM_or_PolyOrder
        best_vals, covar = curve_fit(gaussian, xdata, ydata, p0=[Imax, x0, FWHM, A0]) #5 p0=[amp,cen,wid]
        yfit=gaussian(xfit,*best_vals) 
        FWHM=2.355*best_vals[2]
        center=best_vals[1]
        print('Amplitude: ',best_vals[0])
    elif fct == 'lorz':
        FWHM=FWHM_or_PolyOrder
        best_vals, covar = curve_fit(lorentzian, xdata, ydata, p0=[Imax, x0, FWHM, A0]) #5 p0=[amp,cen,wid]
        yfit=lorentzian(xfit,*best_vals) 
        FWHM=2.355*best_vals[2]
        center=best_vals[1]
        print('Amplitude: ',best_vals[0])
    elif fct == 'erf' or fct == 'box':
        FWHM=FWHM_or_PolyOrder
        xmax=np.max(xdata)
        xmin=np.min(xdata)
        amp=np.mean(ydata)
        x0=np.mean(xdata)
        ymin=np.min(ydata)
        if fct == 'erf':
            yfirst=ydata[np.where(xdata == xmin)][0]
            ylast =ydata[np.where(xdata == xmax)][0]
            if yfirst-ylast >0: amp=-abs(amp) #amp <0 step down
            else: amp = abs(amp)  #amp >0 step up
            p0=[amp, ymin, x0, FWHM]
            popt,pcor = curve_fit(step, xdata, ydata, p0=p0)
            yfit=step(xfit,*popt) 
            FWHM=popt[3]
            center=popt[2]
        elif fct == 'box':
            p0=[amp, x0, FWHM, A0]
            popt,pcor = curve_fit(box, xdata, ydata, p0=p0)
            yfit=box(xfit,*popt) 
            FWHM=popt[2]
            center=popt[1]
    elif fct == 'poly':
        PolyRank=FWHM_or_PolyOrder
        coefs = poly.polyfit(xdata, ydata, PolyRank)
        yfit = poly.polyval(xfit, coefs)
        result=''
        for i in list(range(len(coefs))):
            result=result+'a'+str(i)+' = '+str(round(coefs[i],3))+'\n'
        print(result)
        center=coefs
    
    if fct != 'poly':
        print('Center: ',center)
        print('FWHM: ',FWHM)
        print('\n')
    

    if graph == True:
        center=round(center,4)
        fig,(ax1)=plt.subplots(1,1)
        fig.set_size_inches(5,4)
        ax1.plot(xdata,ydata,label='mda #'+str(scannum),marker=marker)
        ax1.plot(xfit,yfit,label='fit @ '+str(center)[0:6])
        ax1.set(ylabel=y_name)
        ax1.set(xlabel=x_name)
        ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax1.grid(color='lightgray', linestyle='-', linewidth=0.5)
        ax1.legend(shadow=True, title=title, fancybox=True)
    
    return center


def fit_centroid(n):
    fit_mda(n,25,0,'poly',graph=False)

def centroid_avg(scannum,det=25):   # det = detector number for centroid position
    avg=round(fit_mda_poly(scannum,det,0,graph=None)[0],2)
    return avg


def fit_mda_gauss(scannum,det,FWHM,xrange=None,title='',marker='x',graph=1, filepath=None,prefix=None,scanIOC=None):
    fct='gauss'
    fit_mda(scannum,det,FWHM,fct,xrange,title,marker,graph, filepath,prefix,scanIOC)


def fit_mda_lorz(scannum,det,FWHM,xrange=None,title='',marker='x',graph=1, filepath=None,prefix=None,scanIOC=None):
    fct='lorz'
    fit_mda(scannum,det,FWHM,fct,xrange,title,marker,graph, filepath,prefix,scanIOC)


def fit_mda_erf(scannum,det,FWHM,xrange=None,title='',marker='x',graph=1, filepath=None,prefix=None,scanIOC=None):
    fct='erf'
    fit_mda(scannum,det,FWHM,fct,xrange,title,marker,graph, filepath,prefix,scanIOC)

def fit_mda_poly(scannum,det,PolyRank,xrange=None,title='',marker='x',graph=1, filepath=None,prefix=None,scanIOC=None):
    fct='poly'
    fit_mda(scannum,det,PolyRank,fct,xrange,title,marker,graph, filepath,prefix,scanIOC)


def fit_mda_box(scannum,det,FWHM,xrange=None,title='',marker='x',graph=1, filepath=None,prefix=None,scanIOC=None):
    fct='box'
    fit_mda(scannum,det,FWHM,fct,xrange,title,marker,graph, filepath,prefix,scanIOC)


    
    
    
    
    
    
###############################################################################################
######################################### hkl ###################################
###############################################################################################



def plot_hkl(mydata,n,x,y,title='',nlabel=''):
    """mydata = instance of mdaFile object
    n = scan number or [scan number1, ..., scan number N]
    x = h, k or l index (D##)
    d = detector index (D##)
    """
    if type(n) == int:
        maxpts=mydata.dim1[n].curr_pt
        x_index=mydata.dim1[n].kappaDet()[x][0]
        y_index=mydata.dim1[n].kappaDet()[y][0]
        x_name =mydata.dim1[n].d[x_index].desc
        y_name =mydata.dim1[n].kappaDet()[y][2]
        plt.plot(mydata.dim1[n].d[x_index].data[:maxpts],mydata.dim1[n].d[y_index].data[:maxpts],label='scan #'+str(n)+' '+nlabel)
    elif type(n) == list:
        for i,j in enumerate(n): 
            maxpts=mydata.dim1[j].curr_pt
            x_index=mydata.dim1[j].kappaDet()[x][0]
            y_index=mydata.dim1[j].kappaDet()[y][0]
            x_name =mydata.dim1[j].d[x_index].desc
            y_name =mydata.dim1[j].kappaDet()[y][2]
            try:
                curve_label='scan #'+str(j)+' '+nlabel[i]
            except:
                curve_label='scan #'+str(j)
            plt.plot(mydata.dim1[j].d[x_index].data[:maxpts],mydata.dim1[j].d[y_index].data[:maxpts],label=curve_label)
    else:
        print('n must be a single scan (int) or a list of scan')
        return
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.locator_params(tight=None,nbins=5,axis='x')
    plt.legend(ncol=1, shadow=True, title=title, fancybox=True,loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(color='lightgray', linestyle='-', linewidth=0.5)
    plt.show()

    
def fit_hkl(mydata,n,pos,det,FWHM,fct='gauss',xrange=None,title='',marker='x',graph=1):
    """
    mydata = instance of the class mdaFile
         n = scan number
       pos = D## for positioner
       det = D## for detector
      FWHM = width estimate
       fct = 'gauss' or 'lorz'
    xrange = [x1,x2] subrange for fit 
    """


    x_index=mydata.dim1[n].kappaDet()[pos][0]
    y_index=mydata.dim1[n].kappaDet()[det][0]
    x_name =mydata.dim1[n].d[x_index].desc
    y_name =mydata.dim1[n].kappaDet()[det][2]

    x=mydata.dim1[n].d[x_index].data
    y=mydata.dim1[n].d[y_index].data

    def closest(lst, K):
        return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))] 

    if xrange is not None:
        x1=closest(x,xrange[0])
        x2=closest(x,xrange[1])
        xrange=[np.where(x==x1),np.where(x==x2)]
        xmin=min(xrange)
        xmax=max(xrange)
        xmin=min(xrange[0][0])
        xmax=max(xrange[1][0])
        xcrop=x[xmin:xmax]
        ycrop=y[xmin:xmax]
        xdata = np.array(xcrop)
        ydata = np.array(ycrop)
    else:
        xdata = np.array(x)
        ydata = np.array(y)
    
    Imax=np.max(ydata)
    x0=xdata[np.where(ydata==Imax)][0]
    
    nPt_gaus=200
    xfit =  np.linspace(min(x),max(x), nPt_gaus)


    def lorentzian( x, amp, x0, gam ):
        return amp * gam**2 / ( gam**2 + ( x - x0 )**2)
    
    def gaussian(x, amp, cen, wid):
        return amp*np.exp(-np.power(x - cen, 2.) / (2 * np.power(wid, 2.)))
    
    if fct == 'gauss':
        best_vals, covar = curve_fit(gaussian, xdata, ydata, p0=[Imax, x0, FWHM]) #5 p0=[amp,cen,wid]
        yfit=gaussian(xfit,*best_vals) 
    elif fct == 'lorz':
        best_vals, covar = curve_fit(lorentzian, xdata, ydata, p0=[Imax, x0, FWHM]) #5 p0=[amp,cen,wid]
        yfit=lorentzian(xfit,*best_vals)
    else:
        print('Not a valid function: fct = "gauss" or fct = "lorz"')
        return

    FWHM=2.355*best_vals[2]
    center=best_vals[1]
    print('Amplitude: ',best_vals[0])
    print('Center: ',center)
    print('FWHM: ',FWHM)
    
    if graph is not None:
        fig,(ax1)=plt.subplots(1,1)
        fig.set_size_inches(5,4)
        ax1.plot(xdata,ydata,label='data',marker=marker)
        ax1.plot(xfit,yfit,label='fit @ '+str(center)[0:6])
        ax1.set(ylabel=y_name)
        ax1.set(xlabel=x_name)
        ax1.locator_params(tight=None,nbins=5,axis='x')
        ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax1.grid(color='lightgray', linestyle='-', linewidth=0.5)
        ax1.legend(shadow=True, title=title, fancybox=True)
    
    return center


def plot_latest(d=None,norm=None):
    if plotIOC() == "ARPES":
        d=16
    elif plotIOC() == 'Kappa':
        dets=[34,33,35]
        d=dets[caget('29idKappa:det:set')]
    n=mda_GetLastFileNum()
    plot_mda(n,d,norm=norm,marker='+',save=True,title='mda_'+str(LastMDA()))

def fit_d4(n=None):  
    if n==None:
        n=mda_GetLastFileNum()  
    d4=fit_mda(n,34,0.1,'gauss',title='mda_'+str(n).zfill(4))  
    return round(d4,3)

def fit_d3(n=None):  
    if n==None:
        n=mda_GetLastFileNum()  
    d3=fit_mda(n,33,3,'gauss',title='mda_'+str(n).zfill(4))  
    return round(d3,3)

def fit_z(n=None,d=33):
    if n==None:
        n=mda_GetLastFileNum()  
    z=fit_mda(n,d,500,'erf',title='mda_'+str(n).zfill(4))  
    return round(z,0)
    


    
     
def Find_kth_zero(th_0,th_180):
    """
    th_0 is the motor position for specular near zero 
    th_180 is motor position for spectular near 180 
    """
    
    Offset =0.5*(180 - th_180 - th_0)

    print("with motor at 0, the actual value is ",Offset)
    

# def read_hkl_old(n,user=None,run=None):
#     h=[]
#     k=[]
#     l=[]
#     if user is None: user=mda_CurrentUser()
#     if run is None: run=mda_CurrentRun()
#     with open('/home/beams/29IDUSER/Documents/User_Folders/'+user+'/hkl/scan_'+str(n)+'.txt') as csv_file:    
#         csv_f=csv.reader(csv_file,delimiter=',')
#         for row in csv_f:
#             if not row[0]=='h':
#                 h.append(float(row[0]))
#                 k.append(float(row[1]))
#                 l.append(float(row[2]))
#     return np.array(h), np.array(k), np.array(l)

# def plot_hkl_old(mydata,n1,n2=None,which='h',det=39,cropMin=None,cropMax=None,title=''):
#     D=mydata.kappaDet(n1)
#     d=D[det][1]
#     if which == 'h':
#         m = 0
#     if which == 'k':
#         m = 1
#     if which == 'l':
#         m = 2

#     plt.figure(num=None, figsize=(6, 6), dpi=80, facecolor='w', edgecolor='k')
#     if n2 == None:
#         n=n1
#         #dif=len(read_hkl(n)[m])-mydata.dim1[n].curr_pt
#         if cropMax is not None:
#             x=read_hkl(n)[m][cropMin:-cropMax]
#         else:
#             x=read_hkl(n)[m][cropMin:]

#         plt.plot(x,mydata.dim1[n].d[d].data,label='scan #'+str(n),marker='+')
#     else:
#         for n in range(n1,n2+1): 
#             #dif=len(read_hkl(n)[m])-mydata.dim1[n].curr_pt
#             if cropMax is not None:
#                 x=read_hkl(n)[m][cropMin:-cropMax]
#             else:
#                 x=read_hkl(n)[m][cropMin:]

#             plt.plot(x,mydata.dim1[n].d[d].data,label='scan #'+str(n),marker='+')
#     plt.xlabel(which)
#     plt.ylabel(D[det][-1])
#     plt.legend(ncol=1, shadow=True, title=title, fancybox=True,loc='center left', bbox_to_anchor=(1, 0.5))
#     #plt.title(title)
#     plt.grid(color='lightgray', linestyle='-', linewidth=0.5)
#     plt.show()



# def fit_hkl_gauss_old(scannum,det,FWHM,which='l',cropMin=None,cropMax=None,xrange=None,title='',marker='x',graph=1):
#     motor,y,motor_name,y_name=mda_1D(scannum,det,1,0)
#     if which == 'h':
#         m = 0
#     if which == 'k':
#         m = 1
#     if which == 'l':
#         m = 2
#     if cropMax is not None:
#         x=read_hkl(scannum)[m][cropMin:-cropMax]
#     else:
#         x=read_hkl(scannum)[m][cropMin:]
#     def closest(lst, K):
#         return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))] 

#     if xrange is not None:
#         x1=closest(x,xrange[0])
#         x2=closest(x,xrange[1])
#         xrange=[np.where(x==x1),np.where(x==x2)]
#         xmin=min(xrange)
#         xmax=max(xrange)
#         xmin=min(xrange[0][0])
#         xmax=max(xrange[1][0])
#         xcrop=x[xmin:xmax]
#         ycrop=y[xmin:xmax]
#         xdata = np.array(xcrop)
#         ydata = np.array(ycrop)
#     else:
#         xdata = np.array(x)
#         ydata = np.array(y)
    
#     Imax=np.max(ydata)
#     x0=xdata[np.where(ydata==Imax)][0]
    
#     nPt_gaus=200
#     xfit =  np.linspace(min(x),max(x), nPt_gaus)
    
#     def gaussian(x, amp, cen, wid):
#         return amp*np.exp(-np.power(x - cen, 2.) / (2 * np.power(wid, 2.)))
    
#     best_vals, covar = curve_fit(gaussian, xdata, ydata, p0=[Imax, x0, FWHM]) #5 p0=[amp,cen,wid]
#     yfit=gaussian(xfit,*best_vals) 
#     FWHM=2.355*best_vals[2]
#     center=best_vals[1]
#     print('Amplitude: ',best_vals[0])
#     print('Center: ',center)
#     print('FWHM: ',FWHM)
    
#     if graph is not None:
#         fig,(ax1)=plt.subplots(1,1)
#         fig.set_size_inches(5,4)
#         ax1.plot(xdata,ydata,label='data',marker=marker)
#         ax1.plot(xfit,yfit,label='fit @ '+str(center)[0:6])
#         ax1.set(ylabel=y_name)
#         ax1.set(xlabel=which)
#         ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#         ax1.grid(color='lightgray', linestyle='-', linewidth=0.5)
#         ax1.legend(shadow=True, title=title, fancybox=True)
    
#     return center
    
    
###############################################################################################
######################################### Start Of The Week ###################################
###############################################################################################

def plot_StartOfTheWeek_DetDict(branch):
    """
    returns the detectors for a given branch
    
    """
    DetDict={'c':(9,7,8,15),'d':(9,7,8,14)}
    return DetDict[branch]

def StartOfTheWeek_plot(branch,FirstScanNum,**kwargs):
    """
    Plots the data from StartOfTheWeek
    
    branch is used to set the detectors
        detCA4,detH,detV,detDiode=plot_StartOfTheWeek_Det(branch)
        
    FirstScanNum is based on slit1A
        slit1A-H = FirstScanNum
        slit1A-V = FirstScanNum + 1
        wire1-H = FirstScanNum + 2
        wire1-V = FirstScanNum + 3
        monoVslit = FirstScanNum + 4/5/6/7
        flux = FirstScanNum + 8
        
    Slit 1A and Wire scans: determine ID steering and Slit1A position
    Scan_MonoVsSlit: determine the steering from M0/M1 
        by default the full range is plotted (pnt_first=0, pnt_last=inf)
        refine the plot via
            plot_StartOfTheWeek_mono(branch,FirstScanNum,pnt_first,pnt_last)

    kwargs: 
        filepath = None,  uses current mda filepath unless specified
              e.g. user : filepath='/net/s29data/export/data_29idc/2018_2/UserName/mda/'
              e.g. staff: filepath='/net/s29data/export/data_29idb/2018_2/mda/'
        prefix = None, uses current mda prefix unless specified 
        scanIOC = None, uses plotIOC() unless specified
        
        scanType = ['slit1','wire','flux','monoVslit'], full set by default
        
        ref_firstScanNum to plot reference spectra
        ref_branch = branch, unless specified
        ref_filepath = filepath, unless specified
        ref_prefix = prefix, unless specified

    steering out => moves beam more positive (20 urad ~ 0.5 mm)
    steering up  => moves beam more positive (20 urad ~ 0.5 mm)
    
figure, axes = plt.subplots(nrows=2, ncols=2)
axes[0, 0].plot(x, y)
    """
    kwargs.setdefault('scanIOC',plotIOC())
    
    kwargs.setdefault('filepath',mda_CurrentDirectory(kwargs['scanIOC']))
    kwargs.setdefault('prefix',kwargs['scanIOC']+"_")
    
    kwargs.setdefault('scanType',['slit1','wire','flux','monoVslit'])
    
    kwargs.setdefault('ref_firstScanNum',None)
    kwargs.setdefault('ref_fpath',None)
    kwargs.setdefault('ref_branch',branch)
    kwargs.setdefault('ref_filepath',mda_CurrentDirectory())
    kwargs.setdefault('ref_prefix',plotIOC()+"_")
    
    kwargs.setdefault('debug',False)

    
    scanNum=FirstScanNum
    ref_firstScanNum=kwargs['ref_firstScanNum']
    detCA4,detH,detV,detDiode=plot_StartOfTheWeek_DetDict(branch)
    

    if 'slit1' in kwargs['scanType']:
        if kwargs['debug']:
            print('slit1')
        plot_StartOfTheWeek_slit1A(branch,scanNum,**kwargs)
        scanNum+=2
        if kwargs['ref_firstScanNum'] is not None:
            kwargs.update({'ref_firstScanNum':ref_firstScanNum+2})
            
            
    if 'wire' in kwargs['scanType']:
        if kwargs['debug']:
            print('wire')
        plot_StartOfTheWeek_wire(branch,scanNum,**kwargs)
        scanNum+=2
        if kwargs['ref_firstScanNum'] is not None:
            ref_firstScanNum+=2
            kwargs.update({'ref_firstScanNum':ref_firstScanNum+2})
            
    if 'monoVslit' in kwargs['scanType']:
        d=IEXdata(scanNum,path=kwargs['filepath'],prefix=kwargs['prefix'],q=1)
        if d.mda[scanNum].header.all['rank']<2:  #quick
            print(scanNum, detDiode)
            V2=fit_mda(scanNum,detDiode,1,'gauss',filepath=kwargs['filepath'],prefix=kwargs['prefix'],scanIOC=kwargs['scanIOC'],title='2V - mda_'+str(scanNum))
            H2=fit_mda(scanNum+1,detDiode,1,'gauss',filepath=kwargs['filepath'],prefix=kwargs['prefix'],scanIOC=kwargs['scanIOC'],title='2H - mda_'+str(scanNum+1))
            V1=fit_mda(scanNum+2,detDiode,1,'gauss',filepath=kwargs['filepath'],prefix=kwargs['prefix'],scanIOC=kwargs['scanIOC'],title='1V - mda_'+str(scanNum+2))
            H1=fit_mda(scanNum+3,detDiode,1,'gauss',filepath=kwargs['filepath'],prefix=kwargs['prefix'],scanIOC=kwargs['scanIOC'],title='1H - mda_'+str(scanNum+3))
            plt.show()
            print('\nFit positions:')
            print(f"V2={round(V2,3)}")
            print(f"H2={round(H2,3)}")
            print(f"V1={round(V1,3)}")
            print(f"H1={round(H1,3)}")
            
        else:  #MonoVsSlit
            print('\n# To plot the Mono vs Slit data use:\n')
            print('\tplot_MonoVsSlit(\"'+branch+'\",'+str(scanNum)+','+str(detDiode)+',0,inf)'+'\t#2V')
            print('\tplot_MonoVsSlit(\"'+branch+'\",'+str(scanNum+1)+','+str(detDiode)+',0,inf)'+'\t#2H')
            print('\tplot_MonoVsSlit(\"'+branch+'\",'+str(scanNum+2)+','+str(detDiode)+',0,inf)'+'\t#1V')
            print('\tplot_MonoVsSlit(\"'+branch+'\",'+str(scanNum+3)+','+str(detDiode)+',0,inf)'+'\t#1H')
            print('\n#  (pnt_first,pnt_last0=(0,inf) => plots all')
            print('#  select specific first/last to refine.')

        print('\nREMEMBER to update slit center using: \tupdate_slit_dict()' )
        scanNum+=4
        if kwargs['ref_firstScanNum'] is not None:
            kwargs.update({'ref_firstScanNum':ref_firstScanNum+4})
        
    if 'flux' in kwargs['scanType']:
        if kwargs['debug']:
            print('flux')
        plot_StartOfTheWeek_flux(branch,scanNum,**kwargs)

def plot_StartOfTheWeek_slit1A(branch,scanNum,**kwargs):
    """
    plots the slit1A scans
    scanNum = slit1A-H scanNum
    """   
    kwargs.setdefault('filepath',mda_CurrentDirectory())
    kwargs.setdefault('prefix',plotIOC()+"_")
    kwargs.setdefault('scanIOC',plotIOC())
    kwargs.setdefault('plotType',['slit1','wire','flux','monoVslit'])
    kwargs.setdefault('ref_firstScanNum',None)
    kwargs.setdefault('ref_fpath',None)
    kwargs.setdefault('ref_branch',branch)
    kwargs.setdefault('ref_filepath',mda_CurrentDirectory())
    kwargs.setdefault('ref_prefix',plotIOC()+"_")
    kwargs.setdefault('debug',False)

    detCA4,detH,detV,detDiode=plot_StartOfTheWeek_DetDict(branch)
    ref_detCA4,ref_detH,ref_detV,ref_detDiode=plot_StartOfTheWeek_DetDict(kwargs['ref_branch']) 
    
    figure, axes = plt.subplots(nrows=1, ncols=2,figsize=(10,3))
    for i,t in enumerate(['1H center scan','1V center scan']):
        d=IEXdata(scanNum,path=kwargs['filepath'],prefix=kwargs['prefix'],q=1)
        axes[i].plot(d.mda[scanNum].det[detCA4].scale['x'], d.mda[scanNum].det[detCA4].data,marker='x',label=str(scanNum))
        scanNum+=1
        if kwargs['ref_firstScanNum'] is not None:
            ref_scanNum=kwargs['ref_firstScanNum']+i
            ref_d=IEXdata(ref_scanNum,path=kwargs['ref_filepath'],prefix=kwargs['ref_prefix'],q=1)
            axes[i].plot(ref_d.mda[ref_scanNum].det[ref_detCA4].scale['x'], ref_d.mda[ref_scanNum].det[ref_detCA4].data,marker='x',label=str(ref_scanNum))
        axes[i].grid(color='lightgray', linestyle='-', linewidth=0.5)
        axes[i].title.set_text(t)
        axes[i].legend()
    plt.show()
    print("\nsteering out => move beam more positive (10 urad ~ 0.25 mm)")
    print("steering up  => move beam more positive (10 urad ~ 0.25 mm)")
    print("\nTo fit beam position use:\n")
    print("detCA4,detH,detV,detDiode=plot_StartOfTheWeek_DetDict('"+kwargs['ref_branch']+"')")
    print("fit_mda("+str(scanNum-2)+",detCA4,1,'gauss',xrange=(-1,1))")
    print("fit_mda("+str(scanNum-1)+",detCA4,1,'gauss',xrange=(-1,1))\n")

    
def plot_StartOfTheWeek_wire(branch,scanNum,**kwargs):
    """
    plots the wire scans
    scanNum = wire-H scanNum
    """   
    kwargs.setdefault('filepath',mda_CurrentDirectory())
    kwargs.setdefault('prefix',plotIOC()+"_")
    kwargs.setdefault('scanIOC',plotIOC())
    kwargs.setdefault('plotType',['slit1','wire','flux','monoVslit'])
    kwargs.setdefault('ref_firstScanNum',None)
    kwargs.setdefault('ref_fpath',None)
    kwargs.setdefault('ref_branch',branch)
    kwargs.setdefault('ref_filepath',mda_CurrentDirectory())
    kwargs.setdefault('ref_prefix',plotIOC()+"_")
    kwargs.setdefault('debug',False)

    detCA4,detH,detV,detDiode=plot_StartOfTheWeek_DetDict(branch)
    ref_detCA4,ref_detH,ref_detV,ref_detDiode=plot_StartOfTheWeek_DetDict(kwargs['ref_branch']) 
    
    figure, axes = plt.subplots(nrows=1, ncols=2,figsize=(10,3))
    for i,t in enumerate(['H-wire','V-wire']):
        d=IEXdata(scanNum,path=kwargs['filepath'],prefix=kwargs['prefix'],q=1)
        #niceplot(d.mda[scanNum].det[DetDict[branch][1+i]],marker='x',label=str(scanNum))
        detNum=plot_StartOfTheWeek_DetDict(branch)[1+i]
        axes[i].plot(d.mda[scanNum].det[detNum].scale['x'], d.mda[scanNum].det[detNum].data,marker='x',label=str(scanNum))
        scanNum+=1
        if kwargs['ref_firstScanNum'] is not None:
            ref_scanNum=kwargs['ref_firstScanNum']+i
            ref_d=IEXdata(ref_scanNum,path=kwargs['ref_filepath'],prefix=kwargs['ref_prefix'],q=1)
            #niceplot(ref_d.mda[ref_scanNum].det[DetDict[kwargs["ref_branch"]][1+i]],marker='x',label=str(ref_scanNum))
            detNum=plot_StartOfTheWeek_DetDict(kwargs["ref_branch"])[1+i]
            axes[i].plot(ref_d.mda[ref_scanNum].det[detNum].scale['x'], ref_d.mda[ref_scanNum].det[detNum].data,marker='x',label=str(ref_scanNum))
        axes[i].grid(color='lightgray', linestyle='-', linewidth=0.5)
        axes[i].title.set_text(t)
        axes[i].legend()
    plt.show()    
        
def plot_StartOfTheWeek_flux(branch,scanNum,**kwargs):
    """
    plots the wire scans
    scanNum = wire-H scanNum
    """   
    kwargs.setdefault('filepath',mda_CurrentDirectory())
    kwargs.setdefault('prefix',plotIOC()+"_")
    kwargs.setdefault('scanIOC',plotIOC())
    kwargs.setdefault('plotType',['slit1','wire','flux','monoVslit'])
    kwargs.setdefault('ref_firstScanNum',None)
    kwargs.setdefault('ref_fpath',None)
    kwargs.setdefault('ref_branch',branch)
    kwargs.setdefault('ref_filepath',mda_CurrentDirectory())
    kwargs.setdefault('ref_prefix',plotIOC()+"_")
    kwargs.setdefault('debug',False) 

    detCA4,detH,detV,detDiode=plot_StartOfTheWeek_DetDict(branch)
    ref_detCA4,ref_detH,ref_detV,ref_detDiode=plot_StartOfTheWeek_DetDict(kwargs['ref_branch'])     
    
    for t in ['ID peak @ 500eV']:
        d=IEXdata(scanNum,path=kwargs['filepath'],prefix=kwargs['prefix'],q=1)
        niceplot(d.mda[scanNum].det[detDiode],marker='x',label=str(scanNum))
        scanNum+=1
        if kwargs['ref_firstScanNum'] is not None:
            ref_scanNum=kwargs['ref_firstScanNum']+2
            ref_d=IEXdata(ref_scanNum,path=kwargs['ref_filepath'],prefix=kwargs['ref_prefix'],q=1)
            niceplot(ref_d.mda[ref_scanNum].det[ref_detDiode],marker='x',label=str(ref_scanNum))
            ref_scanNum+=1
        plt.grid(color='lightgray', linestyle='-', linewidth=0.5)
        plt.title(t)
        plt.legend()
        plt.show()
    
def plot_MonoVsSlit(branch,ScanNum,detDiode,pnt_first,pnt_last,norm=True,filepath=None,prefix=None,scanIOC=None):
    """
    Plots Scan_MonoVsSlit to determine the steering from M0/M1 
        To plot the full range (pnt_first=0, pnt_last=inf)
    plot_StartofWeek_mono(branch,FirstScanNum+4,pnt_first,pnt_last)
    
    filepath: by default plot scans for the current data folder (as defined in plotIOC() ScanRecord SaveData)
    or specified folder path ending with '/':
        e.g. user : filepath='/net/s29data/export/data_29idc/2018_2/UserName/mda/'
        e.g. staff: filepath='/net/s29data/export/data_29idb/2018_2/mda/'
    prefix: by default, uses prefix as defined in ScanRecord ("mda_")
    """
    x,y,z,x_name,y_name,z_name=mda_2D(ScanNum,detDiode,filepath,prefix,scanIOC)
    Which=str(y_name)[10:12]
    if pnt_last is inf:
        pnt_last=len(z)-1
    for i in range(pnt_first,pnt_last+1):
        maxvalue=max(z[i])
        if norm == True:
            plt.plot(x,z[i]/maxvalue,label='#'+str(i)+': '+str(round(y[i],2)))
        else:
            plt.plot(x,z[i],label='#'+str(i)+': '+str(round(y[i],2)))
        plt.legend(bbox_to_anchor=(1, 0), loc='lower left', ncol=2,shadow=True, title="ScanNum: "+str(ScanNum)+"\nSlit-"+Which+" position", fancybox=True)
        #plt.legend(loc='lower left',ncol=2, shadow=True, title="ScanNum: "+str(ScanNum)+"\nSlit-"+Which+" position", fancybox=True) 
        plt.grid(color='lightgray', linestyle='-', linewidth=0.5)
    plt.show()
    
    
###############################################################################################
######################################### IDCalibration New ###################################
###############################################################################################



def id_calibration_fit(first_scan,last_scan,det,PolyRank,**kwargs):
    
    
    #id_calibration_fit(FirstLast[0],FirstLast[1],det,poly_order,mytitle=mytitle,countby=countby,plotMin=plt_min,plotMax=plt_max,plotType=plotType,filepath=filepath,prefix=prefix)
    """
    Fits calibration curves fpr each GRTs and modes included in the data set. Creates 3 dictionnaries:
    
        coefs={[GRT][mode][[breakpoint1,[coefs]],[breakpoint2,[coefs]...}
        xdata= {[GRT][mode][[breakpoint1,[flux curve x axis]],[breakpoint2,[flux curve x axis]...}
        fdata= {[GRT][mode][[breakpoint1,[flux curve y axis]],[breakpoint2,[flux curve y axis]...}
        
    kwargs:
        countby = 1 by default
        mytitle = '' by default
        plotMin =  min x range for plotting (default 250)
        plotMax =  max x range for plotting (default 3000)
        plotType = ['dif,'fit,'flux'], full set by default
        filepath = None,  uses current mda filepath unless specified
              e.g. user : filepath='/net/s29data/export/data_29idc/2018_2/UserName/mda/'
              e.g. staff: filepath='/net/s29data/export/data_29idb/2018_2/mda/'
        prefix = None, uses current mda prefix unless specified 
        scanIOC = None, uses plotIOC() unless specified
    """

    kwargs.setdefault('countby',1)
    kwargs.setdefault('mytitle','')
    #kwargs.setdefault('plotMin',2000)
    kwargs.setdefault('plotMin',225)
    kwargs.setdefault('plotMax',3000)
    kwargs.setdefault('plotType',['fit','dif','flux'])
    kwargs.setdefault('filepath',mda_CurrentDirectory())
    kwargs.setdefault('prefix',mda_CurrentPrefix())

    countby=kwargs['countby']
    mytitle=kwargs['mytitle']
    plotMin=kwargs['plotMin']
    plotMax=kwargs['plotMax']
    filepath=kwargs['filepath']
    prefix=kwargs['prefix']
    

    def calibration_curve(first,last,det,countby,filepath,prefix):
        mono_list=[]
        ID_list=[]
        max_list=[]
        flux_list=[]
        print("filepath = ",filepath)
        print("prefix = ",prefix)
        print("First scan = ",first)
        print("Last scan  = ",last)
        for i in range(first,last,countby):
            x,y,x_name,y_name=mda_1D(i,det,1,0,filepath,prefix)   #mda_1D(ScanNum,DetectorNum,coeff=1,bckg=0,filepath=None,prefix=None,scanIOC=None):
            v,w,v_name,w_name=mda_1D(i, 4 ,1,0,filepath,prefix)
            if y != []:
                n=y.index(max(y))        # finds the peak max intensity index
                e=round(x[n],2)            # finds the corresponding mono energy
                sp=round(w[2]*1000,0)   # reads the ID set point
                mono_list.append(e)
                max_list.append(max(y))
                ID_list.append(sp)
                flux_list.append(ca2flux(max(y),e,p=None))
        return mono_list,ID_list,max_list,flux_list
    
    Mono_max,ID_SP,int_max,flux=calibration_curve(first_scan,last_scan+1,det,countby,filepath,prefix)
    nPt_gaus=200
    x_HR =  np.linspace(Mono_max[0], Mono_max[-1], nPt_gaus)
    
    # Data
    xdata = np.array(Mono_max)
    ydata = np.array(ID_SP)
    zdata = np.array(int_max)
    fdata = np.array(flux)
    # Fitting
    coefs = poly.polyfit(xdata, ydata, PolyRank)
    ffit_HR = poly.polyval(x_HR, coefs)
    ffit_Coarse = poly.polyval(xdata, coefs)
    # Residual
    Dif=np.array(ID_SP)-np.array(ffit_Coarse)
    ratio=(np.array(ID_SP)-np.array(ffit_Coarse))/(np.array(ID_SP)/100)
    # Plot differences
    #print('plotMin = ',str(plotMin))
    if 'dif' in kwargs['plotType']:
        fig = plt.figure(figsize=(12,6))
        plt.plot(Mono_max,ID_SP,marker='x',markersize=5,color='g',linewidth=0,label='data')
        plt.plot(x_HR,ffit_HR,color='b',label='SP-fit')
        plt.plot(xdata,Dif*100,marker='x',color='r',label='Dif x100')
        plt.plot(xdata,ratio*1000,marker='x',color='g',label='Difx1000/ID')
        plt.ylabel('ID SP')
        plt.xlabel('Mono')
        plt.xlim(plotMin,plotMax)
        plt.legend(ncol=2, shadow=True, title=mytitle, fancybox=True)
        plt.grid(linestyle='-', linewidth='0.5', color='grey')
        plt.show()
    # Plot raw data + fit
    if 'fit' in kwargs['plotType']:
        fig = plt.figure(figsize=(12,6))
        a1 = fig.add_axes([0,0,1,1])
        a1.plot(xdata+Dif,zdata,marker='*',markersize=10,color='r',linewidth=0,label='Interpolated ID SP')
        for i in range(first_scan,last_scan+1):
            x,y,x_name,y_name=mda_1D(i,det,1,0,filepath,prefix)
            a1.plot(x,y,color='b')
        a1.set_xlim(plotMin,plotMax)
        a1.set(xlabel='Mono')
        a1.set(ylabel='ID SP')
        a1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        a1.grid(linestyle='-', linewidth='0.5', color='grey')
        plt.legend(ncol=2, shadow=True, title=mytitle, fancybox=True)
        plt.show()
    # Plot flux curves:
    fdata_x=xdata+Dif
    if 'flux' in kwargs['plotType']:
        fig = plt.figure(figsize=(12,6))
        a1 = fig.add_axes([0,0,1,1])
        a1.plot(fdata_x,fdata,color='r',linewidth=1,label='Flux curves')
        a1.set_xlim(plotMin,plotMax)
        a1.set(xlabel='Mono')
        a1.set(ylabel='ID SP')
        a1.set_yscale('log')
        #a1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        a1.grid(linestyle='-', linewidth='0.5', color='grey')
        plt.legend(ncol=2, shadow=True, title=mytitle, fancybox=True)
        plt.show()
    return coefs, fdata_x,fdata 






def read_id_files(first,last,filepath=None,prefix=None,q=True):
    """
    Reads extra PVs
    Return a list of [[ScanNum,ID_SP,grt,mode],...] for mda files between first and last.
    """
    if filepath == None:
        filepath=mda_CurrentDirectory()
    if prefix == None:
        prefix=mda_CurrentPrefix()[:-1]
    elif prefix[-1] == "_":
        prefix=prefix[:-1]
    mydata=mdaFile(first,last,filepath=filepath,prefix=prefix,q=q)
    value=[]
    modeDict={'CW,  RCP':'RCP','CCW, LCP':'LCP','H':'H','V':'V'}
    grtDict={1200:'MEG',2400:'HEG'}
    for mdaNum in mydata.scanList:
        if first<=mdaNum<=last:
            extraPVs=mydata.header[mdaNum].all
            try:
                ID=round(extraPVs['ID29:EnergySet.VAL'][2][0]*1000,2)
                mode=modeDict[extraPVs['ID29:ActualMode'][2]]                # extraPVs return 'CW, RCP'
                grt=grtDict[extraPVs['29idmono:GRT_DENSITY'][2][0]]        # extraPVs return 1200
            except KeyError:
                ID=round(extraPVs[b'ID29:EnergySet.VAL'][2][0]*1000,2)
                mode=modeDict[str(extraPVs[b'ID29:ActualMode'][2])[2:-1]]
                grt=grtDict[extraPVs[b'29idmono:GRT_DENSITY'][2][0]]        # extraPVs return 1200
            if len(value)>0 and value[-1][1:] == [ID,grt,mode]:
                pass
            else:
                value.append([mdaNum,ID,grt,mode])
    return value



def id2num(ID,grt,mode,first=0,last=inf,ignore=[],filepath=None,prefix=None,q=True):   # Not efficient - requires to read all 600 files everytime
    """
    Return ScanNum corresponding to a given ID_SP from ExtraPVs (using mdaFile)
    """
    if filepath == None:
        filepath=mda_CurrentDirectory()
    if prefix == None:
        prefix=mda_CurrentPrefix()[:-1]
    elif prefix[-1] == "_":
        prefix=prefix[:-1]
    ScanNum = 0
    data_list = read_id_files(first,last,filepath,prefix,q)
    data_short=[x for x in data_list if x[2:]==[grt,mode] and x[0] not in ignore]
    step=data_short[1][1]-data_short[0][1]
    precision=int(step/2)
    ID1=ID-precision
    ID2=ID+precision
    ScanNum=[x[0] for x in data_short if ID1<= x[1]<= ID2]
    if len(ScanNum)==0: result=None
    else: result=ScanNum[0]
    return result



def num2id(ScanNum,grt,mode,first=0,last=inf,ignore=[],filepath=None,prefix=None,q=True):  # Not efficient - requires to read all 600 files everytime
    """
    Return ID SP corresponding to a given ScanNum from ExtraPVs (using mdaFile)
    """
    if filepath == None:
        filepath=mda_CurrentDirectory()
    if prefix == None:
        prefix=mda_CurrentPrefix()[:-1]
    elif prefix[-1] == "_":
        prefix=prefix[:-1]
    ID = 0
    data_short=[]
    data_list = read_id_files(first,last,filepath,prefix,q)
    data_short=[x for x in data_list if x[2:]==[grt,mode] and x[0] not in ignore]
    ID=[x[1] for x in data_short if x[0] == ScanNum]
    return ID[0]



def extract_id(first,last,ignore=[],breakpts={'RCP':[600],'H':[400,600],'V':[600],'LCP':[600],'HEG':[],'MEG':[2200,2475]},filepath=None,prefix=None,q=True):
    """
    Breaksdown the info from a calibration files into grt & mode with the corresponding breakpoints (hardcoded):
        [[first, last, 'HEG', 'RCP', [600]],
         [first, last, 'HEG', 'H', [400, 600]]...]  
    """
    if filepath == None:
        filepath=mda_CurrentDirectory()
    if prefix == None:
        prefix=mda_CurrentPrefix()[:-1]
    elif prefix[-1] == "_":
        prefix=prefix[:-1]
    IDlog=read_id_files(first,last,filepath,prefix,q)
    #breakpts={'RCP':[600],'H':[400,600],'V':[600],'LCP':[600],'HEG':[],'MEG':[2200]}
    #breakpts={'RCP':[600],'H':[400,600],'V':[600],'LCP':[600],'HEG':[],'MEG':[2200]}
    #breakpts={'RCP':[600],'H':[600],'V':[600],'LCP':[600],'HEG':[],'MEG':[2200,2475]}  # FR changed H breakpoints, missing low energy scans 06/14/2021
    data=[]
    for grt in ['HEG','MEG']:
        for mode in ['RCP','H','V','LCP']:
            tmp=[x for x in IDlog if x[2:]==[grt,mode] and x[0] not in ignore]
            #print(tmp)
            if len(tmp)>0:
                tmp=[tmp[0][0],tmp[-1][0],grt,mode,breakpts[mode]+breakpts[grt]]
            data.append(tmp)
    return data



def update_id_dict(first,last,det,update_file,ignore,**kwargs):
    
    

    """
    Calculate new calibration curve for a full set of data.'
    But what if the set is not complete? To be addressed by Future Me
    If update_file == True (ONLY):
    \tupdate the ID dictionary '/home/beams22/29IDUSER/Documents/User_Macros/Macros_29id/IEX_Dictionaries/Dict_IDCal.txt'
    """ 

    if update_file == True:
        foo=input('\nAre you sure you want to update the ID calibration dictionary?\n>')
        if foo == 'Y' or foo == 'y' or foo == 'yes'or foo == 'YES' or foo == 'Yes':
            foo = 'yes'
            print('\nWill save new dictionary to: \'/home/beams22/29IDUSER/Documents/User_Macros/Macros_29id/IEX_Dictionaries/Dict_IDCal.txt\'\n')
        else:
            print('\nCalibration curves will not be saved.\n')
    else:
        print('\nCalibration curves will not be saved.\n')

    kwargs.setdefault('countby',1)
    kwargs.setdefault('mytitle','')
    kwargs.setdefault('plotMin',225)
    kwargs.setdefault('plotMax',3000)
    kwargs.setdefault('plotType',['fit','dif','flux'])
    kwargs.setdefault('filepath',mda_CurrentDirectory())
    kwargs.setdefault('prefix',mda_CurrentPrefix())
    kwargs.setdefault('q',True)
    kwargs.setdefault('breakpts',{'RCP':[600],'H':[400,600],'V':[600],'LCP':[600],'HEG':[],'MEG':[2200,2475]})


    breakpts=kwargs['breakpts']
    plotType=kwargs['plotType']
    filepath=kwargs['filepath']
    prefix=kwargs['prefix']
    q=kwargs['q']

    ID_data=extract_id(first,last,ignore,breakpts,filepath,prefix,q)

    ##### Extract parameters:

    mode_dict ={'RCP':0,'LCP':1,'V':2,'H':3}
    id_coef={}
    new_id_function={}
    id_flux={}
    flux_dict={}
    id_energy={}
    energy_dict={}
    for mylist in ID_data:
        if len(mylist) > 0:         # in case of incomplete data set
            tmp0,tmp1,tmp2=[],[],[]
            first_scan=mylist[0]
            last_scan=mylist[1]
            scan_list=[[first_scan,last_scan]]
            countby=1
            #det=15
            #det=33
            poly_order=4
            grt=mylist[2]
            mode=mylist[3]
            breakpts_energy=mylist[4]
            breakpts_scan=[]
            mytitle=mode+' - '+grt 
            
            ## generating list of scans depending on breakpoints:
            if breakpts_energy != []:
                scan_list=[]
                for x in breakpts_energy:
                    breakpts_scan.append(id2num(x,grt,mode,first,last,ignore,filepath,prefix,q))
                breakpts_scan = [i for i in breakpts_scan if i]
                print(breakpts_scan)
                for c,x in  enumerate(breakpts_scan):              # can get the number by extracting step size then /2
                    if c == 0:
                        if x == first_scan: pass
                        else: scan_list.append([first_scan,x])
                    if 0 < c < len(breakpts_scan)-1:
                        scan_list.append([breakpts_scan[c-1],x])
                        scan_list.append([x,breakpts_scan[c+1]])
                    if c == len(breakpts_scan)-1 and (c-1) == 0:
                        scan_list.append([breakpts_scan[c-1],x])
                    if c == len(breakpts_scan)-1:
                        scan_list.append([x,last_scan])
            final_list = [i for n, i in enumerate(scan_list) if i not in scan_list[:n]]     # remove doubles
            energy_list=[]
            for x in final_list:
                ID1=num2id(x[0],grt,mode,first,last,ignore,filepath,prefix,q)
                ID2=num2id(x[1],grt,mode,first,last,ignore,filepath,prefix,q)
                energy_list.append([ID1,ID2])
            energy_list_2=energy_list           # we need the final value for plot purposes (max value)
            if grt == 'HEG':
                energy_list_2[-1][-1]=2500          # but we don't want it to be a cutoff for ID_Calc in dict 
            for (FirstLast,ID,cutoff) in zip(final_list,energy_list,energy_list_2):   
                    plt_min=ID[0]-200
                    plt_max=ID[1]+100
                    if grt=='MEG' and ID[1]>2900:
                        poly_order=5
                        plt_min=ID[0]-200
                    print('det =',det)
                    print('poly_order =',poly_order)
                    
                    subkwargs=kwargs # kwargs contains plotType, filepath, prefix and q
                    newkwargs={'mytitle':mytitle,'countby':countby,"plotMin":plt_min,'plotMax':plt_max}
                    subkwargs.update(newkwargs)
                    #print(newkwargs)
                    #print(FirstLast[0],FirstLast[1])
                    #result,energy_data,flux_data=id_calibration_fit(FirstLast[0],FirstLast[1],det,poly_order,subkwargs)
                    result,energy_data,flux_data=id_calibration_fit(FirstLast[0],FirstLast[1],det,poly_order,mytitle=mytitle,countby=countby,plotMin=plt_min,plotMax=plt_max,plotType=['fit','dif'],filepath=filepath,prefix=prefix)
                    tmp0.append([cutoff[1],result.tolist()])
                    tmp1.append([cutoff[1],energy_data])
                    tmp2.append([cutoff[1],flux_data])
            
            if 2900<tmp0[-1][0]<3000: tmp0[-1][0]=3000
            id_coef[mode_dict[mode]]=tmp0      # dictionary that
            new_id_function[grt]=id_coef      # dictionary containing all the calibration curves forthe data set
            id_energy[mode_dict[mode]]=tmp1      # dictionary that
            energy_dict[grt]=id_energy    
            id_flux[mode_dict[mode]]=tmp2      # dictionary that
            flux_dict[grt]=id_flux

    ##### Read & update old dictionary:




    try:
        id_function=read_dict('Dict_IDCal.txt')
        print(id_function)
        id_function.update(new_id_function)

    except KeyError:
        print("Unable to read previous dictionary")


    if update_file == True and foo == 'yes':
        filepath = "/home/beams22/29IDUSER/Documents/User_Macros/Macros_29id/IEX_Dictionaries/"
        filename = 'Dict_IDCal.txt'

        with open(join(filepath, filename), "a+") as f:   
            f.write('\n======= '+today_plot()+': \n')
            f.write(str(id_function))  
            f.write('\n')
            print('\nWriting dictionary to:',join(filepath, filename))

    if update_file == True and foo == 'yes':
        filepath = "/home/beams22/29IDUSER/Documents/User_Macros/Macros_29id/IEX_Dictionaries/"
        filename = 'Flux_Curves.txt'

        with open(join(filepath, filename), "a+") as f:   
            f.write('\n======= '+today_plot()+': \n')
            f.write('\n----- flux_x:\n')
            f.write(str(energy_dict))  
            f.write('\n----- flux_y:\n')
            f.write(str(flux_dict))  
            f.write('\n')
            print('\nWriting flux curves to:',join(filepath, filename))

    return id_function,energy_dict,flux_dict


def update_slit_dict(**kwargs):
    """
    Interactive function to update the slit position dictionary (Dict_Slit.txt)
    
    **kwargs
        readOnly == False, if true just pring the current slit dictionary
    """
    filepath = "/home/beams22/29IDUSER/Documents/User_Macros/Macros_29id/IEX_Dictionaries/"
    filename ='Dict_Slit.txt'
    
    kwargs.setdefault("readOnly",False)

    print('\n\nWARNING: 2V slit should always be centered at zero ie steer M1 center beam on the grating')
    print('Increasing the roll moves the beam more positive: +0.05 pitch => ~ +0.04 slit position; Use:\n')
    print("Get_Mirror(1);Move_M1('RZ',x)")
    print("energy(500);mono(505);slit(200)")
    print("Scan_NarrowSlit_Go(which='2V',slit_parameters=[0.25, -2, 2, 0.25]) #for MEG; slit_parameters=[0.5, -2, 2, 0.25] for HEG")
    print("Scan_NarrowSlit_Go(which='2V',slit_parameters=[0.50, -2, 2, 0.25]) #for HEG")
    print('Once steering is completed, check 2H position:')
    print("Scan_NarrowSlit_Go(which='2H',slit_parameters=[0.25, -6, 6, 0.25]) ")
    

    try:
        slit_position=read_dict(filename)
        print('\nCurrent dictionary:\n')
        print(slit_position)

    except KeyError:
        print("Unable to read previous dictionary")
        return
    
    if kwargs['readOnly']== True:
        return slit_position
        
    else:
        grt=input('\nWhich grating do you want to update (HEG or MEG) >')
        s2v=input('New Slit 2B-V center >')
        s2h=input('New Slit 2B-H center >')
        s1v=input('New Slit 1A-V center >')
        s1h=input('New Slit 1A-H center >')

        if grt == '' or s1h == '' or s1v == '' or s2h == '' or s2v == '':
            print('\nMissing input. No changes made in file.')
            return

        if grt[0] == '"' or grt[0]=="'": grt = grt[1:4]

    #    new_slit_position={grt.upper():{'S1H':float(s1h),'S1V':float(s1v),'S2H':float(s2h),'S2V':float(s2v)}}   # change order to match scan/fit order
        new_slit_position={grt.upper():{'S2V':float(s2v),'S2H':float(s2h),'S1V':float(s1v),'S1H':float(s1h)}}
        slit_position.update(new_slit_position)


        with open(join(filepath, filename), "a+") as f:
            f.write('\n======= '+today_plot()+': \n')
            f.write(str(slit_position))
            f.write('\n')
            print('\nWriting dictionary to:',join(filepath, filename))

        print('\nUse SetSlit_BL() to update slit positions')
        return slit_position






def read_flux(FileName='Flux_Curves.txt',FilePath="/home/beams22/29IDUSER/Documents/User_Macros/Macros_29id/IEX_Dictionaries/"):
    print('test')
    with open(join(FilePath, FileName)) as f:
        for c,line in enumerate(f.readlines()):
            if line[0] == '=':
                lastdate=line[8:16]
                print(lastdate)
            if line[0] == '-' and line[-2] == 'x':
                axis='x'
                print(axis)
            line_x=line
            if line[0] == '-' and line[-2] == 'y':
                axis='y'
                print(axis)
            line_y=line
        mydict_x=ast.literal_eval(line_x)
        mydict_y=ast.literal_eval(line_y)
    return mydict_x,mydict_y






###############################################################################################
######################################### Object Oriented #####################################
###############################################################################################

class _mdaData(scanDim):
    
    def __init__(self):
        scanDim.__init__(self)
        self.poslist = None
        self.detlist = None
        
    
    def _getDetIndex(self,d):
        """ 
            d = det Num
        """
        D=self.detlist
        if D==[]:
            print('Data contains no detectors. Try higher dimensions: mydata.dim2[n]')
            index=[None]
        else:
            index=[x for x, y in enumerate(D) if y[1] == 'D'+str(d).zfill(2)]
            if index == []:
                print('Detector '+str(d)+' not found.')
                index=[None]
        return index[0]
        

    
    def plt(self,d):
        d=self._getDetIndex(d)
        if d== None:
            return
        x=self.p[0]
        y=self.d[d]
        if self.dim == 2:
            print('This is a 2D scan; use method mydata.img(n,d)')
            for i in range(len(y.data)):
                plt.plot(x.data[i], y.data[i],label=y.fieldName,marker='+')            # crop aborted scans (curr_pt<npts)
        else:   plt.plot(x.data[:self.curr_pt], y.data[:self.curr_pt],label=y.fieldName,marker='+')            # crop aborted scans (curr_pt<npts)
        plt.xlabel(x.name)
        plt.ylabel(y.name)
        plt.legend()
        plt.grid(color='lightgray', linestyle='-', linewidth=0.5)
        
        
        
    def kappaDet(self,q=None):
        if q is not None:
            print('\nUse detIndex for plot: mydata.dim1[n].d[detIndex].data\nIf D=mydata.dim1[n].kappaDet => detIndex=D[detNum][1]\n')
            print('key = (detIndex, detNum, description, pv)')
        det={}
        D=self.detlist
        for (i,j) in zip([32,33,34,35,31,36,37,38,39,41,42,43,44,45,46,47,48,54,55,56,57],['tey','d3','d4','mcp','mesh','tey / mesh','d3 / mesh','d4 / mesh','mcp / mesh','roi1','roi2','roi3','roi4','roi5','<h>','<k>','<l>','tth','th','chi','phi']):
            d=self._getDetIndex(i)
            if d != None:
                det[i]=(d,D[d][1],j,D[d][2])
            else:
                det[i]=None
        return det        
    


        
    
class _mdaHeader:
    def __init__(self):
        self.all = None
        self.sample = None
        self.mono = None
        self.ID = None
        self.energy = None
        self.det = None
        self.motor = None
        self.mirror = None
        self.centroid = None
        self.slit=None
        self.comment=None
        
        


class mdaFile:
    
    
    '''mydata=mdaFile(first=0,last=None,name=datasetName,filepath=None,prefix=None)
    
    /net/s29data/export/data_29idb/2020_3/mda/Kappa_0107.mda is a 1-D file; 1 dimensions read in.
    
    mydata.header[n] = dictionary of 163 scan-environment PVs
    
       usage: mydata.header[n]['sampleEntry'] -> ('description', 'unit string', 'value', 'EPICS_type', 'count')
    
    mydata.dim1[n] = 1D data from '29idKappa:scan1' acquired on Oct 20, 2020 19:06:23:
                    33/33 pts; 1 positioners, 65 detectors
    
       usage: mydata.dim1[n].p[2].data -> 1D array of positioner 1 data
     
    Each scan dimension (i.e., dim1, dim2, ...) has the following fields: 
          time      - date & time at which scan was started: Oct 20, 2020 19:06:23
          name      - name of scan record that acquired this dimension: 29idKappa:scan1
          curr_pt   - number of data points actually acquired: 33
          npts      - number of data points requested: 33
          nd        - number of detectors for this scan dimension: 65
          d[]       - list of detector-data structures
          np        - number of positioners for this scan dimension: 1
          p[]       - list of positioner-data structures
          nt        - number of detector triggers for this scan dimension: 1
          t[]       - list of trigger-info structures
     
    Each detector-data structure (e.g., dim[1].d[0]) has the following fields: 
          desc      - description of this detector
          data      - data list
          unit      - engineering units associated with this detector
          fieldName - scan-record field (e.g., 'D01')
     
    Each positioner-data structure (e.g., dim[1].p[0]) has the following fields: 
          desc          - description of this positioner
          data          - data list
          step_mode     - scan mode (e.g., Linear, Table, On-The-Fly)
          unit          - engineering units associated with this positioner
          fieldName     - scan-record field (e.g., 'P1')
          name          - name of EPICS PV (e.g., 'xxx:m1.VAL')
          readback_desc - description of this positioner
          readback_unit - engineering units associated with this positioner
          readback_name - name of EPICS PV (e.g., 'xxx:m1.VAL')
    '''

    def __init__(self,first=1,last=None,name='mydata',filepath=None,prefix=None,q=False):
        if filepath == None:
            filepath = mda_CurrentDirectory()
        self.path  = filepath
        self._name  = name
        self._first = first
        self._last  = last
        if prefix != None and prefix[-1]=='_':
            self._prefix= prefix[:-1]
        else:
            self._prefix= prefix
            
        self._allFiles = None
        self._allPrefix  = None
        self.loadedFiles = None
        self.scanList  = None
        self.dim1  = None
        self.dim2  = None
        self.dim3  = None
        self.header  = None

        self._extractFiles()
        self._extractData(q)


    #def __str__(self):



    def _extractFiles(self):
        allFiles   = [f for f in listdir(self.path) if isfile(join(self.path, f))]
        loadedFiles= [x for (i,x) in enumerate(allFiles) if allFiles[i].split('.')[-1]=='mda']
        allPrefix = [loadedFiles[i][:loadedFiles[i].find('_')] for (i,x) in enumerate(loadedFiles)]
        scanList = [int(loadedFiles[i][loadedFiles[i].find('_')+1:loadedFiles[i].find('_')+5]) for (i,x) in enumerate(loadedFiles)]
        if self._prefix != None:
            allPrefix=[s for s in allPrefix if s == self._prefix]
            scanList = [int(loadedFiles[i][loadedFiles[i].find('_')+1:loadedFiles[i].find('_')+5]) for (i,x) in enumerate(loadedFiles) if loadedFiles[i][:loadedFiles[i].find('_')] == self._prefix]
            loadedFiles   = [s for s in loadedFiles if s[:s.find('_')] == self._prefix]
        else:
            self._prefix=allPrefix[-1]
            if allPrefix[0]!=allPrefix[-1]:
                print('\nWARNING: Found more than one file prefix: {}, {}.\n\nData with the same scan number will be overwriten in the order they are loaded. \nPlease specify the one you want to load with arg prefix="which".\n\n'.format(allPrefix[0],allPrefix[-1]))
        if self._last == None:
            self._last = self._first
        shortlist  = [i for (i,x) in enumerate(scanList) if self._first<=x<=self._last]  
        self._allFiles = allFiles
        self.loadedFiles = [loadedFiles[i] for i in shortlist]  
        self.scanList  = [scanList[i] for i in shortlist]
        self._allPrefix=[allPrefix[i] for i in shortlist]
        
        
    def _extractData(self,q):
        
        allheader = {}
        alldata1 = {}
        alldata2 = {}
        alldata3 = {}
        
        for (i,mda) in enumerate(self.loadedFiles):
            
            ##### File info:
            
            filename=mda
            filepath=self.path
            #print(filepath)
            num=self.scanList[i]
            #print(num)
            fullpath=join(filepath,filename)
            #print(fullpath)
            data=readMDA(fullpath,useNumpy=True)    # data = scanDim object of mda module
            
            ###### Extract header:

            D0 = _mdaHeader()   # initiate D0 = mdaHeader object
            D1 = _mdaData()
            D2 = _mdaData()
            D3 = _mdaData()
            
            D=[]
            
            for d in range(0,4):
                if d in range(0,len(data)): D.append(data[d])
                else: D.append(None)
            
            # D[0]=data[0]
            # D[1]=data[1]
            # D[2]=None if 1D data, data[2] if 2D data
            # D[3]=None if 2D data, data[3] if 3D data
                
            
            D0.all=D[0]
            
            
            if filename[:5] == 'Kappa':
                try:
                    D0.sample={**{key:value[:3] for key, value in D[0].items() if '29idKappa:m' in key},**{key:value[:3] for key, value in D[0].items() if '29idKappa:Euler' in key},**{key:value[:3] for key, value in D[0].items() if 'LS331' in key}}
                    D0.mirror = {key:value[:3] for key, value in D[0].items() if '29id_m3r' in key}
                    D0.centroid={key:value[:3] for key, value in D[0].items() if 'ps6' in key.lower()}
                    D0.det = {key:value[:3] for key, value in D[0].items() if '29idd:A' in key}
                    detkeys=['29idMZ0:scaler1.TP','29idKappa:m9.RBV','29idKappa:userCalcOut10.OVAL','29iddMPA:C0O','29idKappa:userStringSeq6.STR1','29idd:Unidig1Bo0']
                    for k in detkeys: 
                        if k in D[0]: D0.det[k]=D[0][k][:3]
                    D0.ID={key:value[:3] for key, value in D[0].items() if 'ID29' in key}
                    D0.UB={key:value[:3] for key, value in D[0].items() if 'UB' in key}
                    D0.mono={key:value[:3] for key, value in D[0].items() if 'mono' in key}
                    D0.energy={key:value[:3] for key, value in D[0].items() if 'energy' in key.lower()}
                    D0.motor = {key:value[:3] for key, value in D[0].items() if '29idb:m' in key}
                    D0.slit={key:value[:3] for key, value in D[0].items() if 'slit3d' in key.lower()}
                except:
                    pass
            if filename[:5] == 'ARPES':
                try:
                    #D0.sample={**{key:value[:3] for key, value in D[0].items() if '29idKappa:m' in key},**{key:value[:3] for key, value in D[0].items() if '29idKappa:Euler' in key},**{key:value[:3] for key, value in D[0].items() if 'LS331' in key}}
                    #D0.mirror = {key:value[:3] for key, value in D[0].items() if '29id_m3r' in key}
                    #D0.centroid={key:value[:3] for key, value in D[0].items() if 'ps6' in key.lower()}
                    #D0.det = {key:value[:3] for key, value in D[0].items() if '29idd:A' in key}
                    #detkeys=['29idMZ0:scaler1.TP','29idKappa:m9.RBV','29idKappa:userCalcOut10.OVAL','29iddMPA:C0O','29idKappa:userStringSeq6.STR1','29idd:Unidig1Bo0']
                    #for k in detkeys: 
                    #    if k in D[0]: D0.det[k]=D[0][k][:3]
                    D0.ID={key:value[:3] for key, value in D[0].items() if 'ID29' in key}
                    #D0.UB={key:value[:3] for key, value in D[0].items() if 'UB' in key}
                    D0.mono={key:value[:3] for key, value in D[0].items() if 'mono' in key}
                    D0.energy={key:value[:3] for key, value in D[0].items() if 'energy' in key.lower()}
                    D0.motor = {key:value[:3] for key, value in D[0].items() if '29idb:m' in key}
                    D0.slit={key:value[:3] for key, value in D[0].items() if 'slit3c' in key.lower()}
                except:
                    pass

                try:
                    cmt1=D[0]['29id'+self._prefix+':saveData_comment1'][2]
                    cmt2=D[0]['29id'+self._prefix+':saveData_comment2'][2]
                    if cmt2 != '': D0.comment = cmt1+' - '+cmt2
                    else: D0.comment = cmt1grid
                except:
                    D0.comment = ''
            
            
            ###### Extract data:
            
            DIMS=[D1,D2,D3]
            
            for counter, value in enumerate(DIMS):
                c=counter+1
                if D[c] is not None:
                    value.rank=D[c].rank
                    value.dim=D[c].dim
                    value.npts=D[c].npts
                    value.curr_pt=D[c].curr_pt
                    value.plower_scans=D[c].plower_scans
                    value.name=D[c].name #
                    value.time=D[c].time
                    value.np=D[c].np
                    value.p=D[c].p
                    value.nd=D[c].nd
                    value.d=D[c].d
                    value.nt=D[c].nt
                    value.t=D[c].t
                    value.detlist=[(i,D[c].d[i].fieldName,D[c].d[i].name,D[c].d[i].desc) for i in range(0,D[c].nd)]
                    value.poslist=[(i,D[c].p[i].fieldName,D[c].p[i].name,D[c].p[i].desc) for i in range(0,D[c].np)]
                else:
                    value=None
            
            allheader[num] = D0
            alldata1[num]  = D1
            alldata2[num]  = D2
            alldata3[num]  = D3
            
            d=D.index(None)-1
            if q is False:
                print('Loading {}  as  {}.dim{}[{}]:\n\t\t...{}D data, {}/{} pts; {} positioners, {} detectors'.format(
                    filename,self._name,d,self.scanList[i],D[d].dim,D[d].curr_pt, D[d].npts, D[d].np, D[d].nd))
        
        self.header=allheader
        self.dim1=alldata1
        self.dim2=alldata2
        self.dim3=alldata3
        




    def updateFiles(self,first=0,last=inf,name=None,filepath=None,prefix=None):
        new=mdaFile(first,last,name,filepath,prefix)
        self.loadedFiles=list(dict.fromkeys(self.loadedFiles+new.loadedFiles))
        self._allFiles=list(dict.fromkeys(self._allFiles+new._allFiles))              # merging the 2 list and removing duplicates
        self.scanList=list(dict.fromkeys(self.scanList+new.scanList))
        self._allPrefix=list(dict.fromkeys(self._allPrefix+new._allPrefix))
        self.dim1.update(new.dim1)
        self.dim2.update(new.dim2)
        self.dim3.update(new.dim3)
        self.header.update(new.header)
        return self
    
    

    def plt(self,*argv):
        if self.dim2[argv[0]].dim == 0:              #1D scan
            for index,arg in enumerate(argv):
                if index %2 !=0:
                    pass
                else:
                    n=arg
                    d=argv[index+1]
                    d=self.dim1[n]._getDetIndex(d)
                    x=self.dim1[n].p[0]
                    y=self.dim1[n].d[d]
                    plt.plot(x.data[:self.dim1[n].curr_pt], y.data[:self.dim1[n].curr_pt],label='mda #'+str(n)+' - '+y.fieldName,marker='+')
            plt.xlabel(x.name)
            plt.ylabel(y.name+' - ('+y.fieldName+')')
            plt.legend()
            plt.grid(color='lightgray', linestyle='-', linewidth=0.5)
            plt.show()
        elif self.dim2[argv[0]].dim == 2:              # 2D scan 
            for index,arg in enumerate(argv):
                if index %2 !=0:
                    pass
                else:
                    n=arg
                    d=argv[index+1]
                    d=self.dim2[n]._getDetIndex(d)
                    if d == None:
                        return
                    x=self.dim2[n].p[0]
                    y=self.dim1[n].p[0]
                    z=self.dim2[n].d[d]
                    zlim=self.dim2[n].curr_pt
                    fig, ax0 = plt.subplots()
                    img = ax0.imshow(z.data[:zlim],cmap='gnuplot', interpolation = 'nearest', extent = [min(x.data[0]), max(x.data[0]), min(y.data),max(y.data)], aspect = 'auto')
                    fig.colorbar(img)
                    plt.title(z.name+' - ('+z.fieldName+')')
                    ax0.set_xlabel(x.name)
                    ax0.set_ylabel(y.name)
                    plt.show()








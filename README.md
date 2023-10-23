# Soft-X-ray-Quantum-Efficiency-of-a-Commercial-CMOS-Imaging-Sensor
Using the Advanced Photon Source at Argonne National Laboratory, we measure the soft X-ray quantum efficiency of a commercially available backside-illuminated CMOS sensor


Unzip NumFrameCountDict.7z to get NumFrameCountDict.pkl
Unzip AlignMosaicGD4withPrintedOutputs.7z to get AlignMosaicGD4withPrintedOutputs.pdf, which is a .pdf of AlignMosaicGD4.ipynb with the outputs still displayed. AlignMosaicGD4.ipynb was too large to upload with all the outputs still displayed.

Running the following Jupyter notebooks in the order listed will reproduce our data analysis for this paper.

1.)  FindSteadyStateFrames1.ipynb
	Identifies the steady state frames and sorts them into the appropriate folders

2.)  MakeMasterHotsArrayAndDarks1.ipynb
	Makes Master Hot Pixel Array Master Dark Frames 
	(For the non-gain data, adu, MeanCS_Dark.csv)

3.)  GainDetermination1p1.ipynb
	Makes the composite open shutter darks for the gain datasets and then finds 
	the gain

4.)  GainDeterminationFigure1.ipynb
	Draws the figure for gain determination (for the omitted appendix section)

5.)  MakeMeanAndSumTiles4.ipynb
	Makes Mean and Sum Tiles 
	(and MeanExpTimeDict.pkl, NumFrameCountDict.pkl, and FrameCountDict.pkl)

6.)  AlignMosaicGD4.ipynb
	Finds the finalized mosaic alignment

7.)  MosaicFluxAndError2.ipynb
	Constructs the final Mosaic (and Figure), determines the finalized e- flux,
	and determines the associated uncertainty.

8.)  PhotodiodeCalibration1.ipynb
	Reads .mda and does photodiode calibration at 490.0 and 551.2 eV

9.)  PhotodiodeResponsivityCalculation1.ipynb
	Calculates responsivity of AXUV100 photodiode (UPD) and uncertainty 
	using information from NIST Paper

10.) PhotonFlux_QE_And_Uncertainty1.ipynb
	Determines final photon flux and uncertainty, then calculates final QE 
	and uncertainty

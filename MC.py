# coding: utf-8
import vtk

import numpy as np

from vtk.util import numpy_support
from vtk.util.numpy_support import vtk_to_numpy

from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = 'STZhongsong'
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12   
plt.rcParams['axes.unicode_minus'] = False

dicomPath = "./vhm_head/"
figPath = 'E:/Documents/Study/VISC/Transfer function/figs'

# Tools
def vtkImageToNumPy(image, pixelDims):
	pointData = image.GetPointData()
	arrayData = pointData.GetArray(0)
	ArrayDicom = numpy_support.vtk_to_numpy(arrayData)
	ArrayDicom = ArrayDicom.reshape(pixelDims, order='F')
	
	return ArrayDicom

def plotHeatmap(array, name="plot"):
	plt.figure()
	im = plt.imshow(array, cmap=plt.cm.hot_r)
	plt.title(name)
	plt.colorbar(im)
	plt.savefig('%s/%s.pdf'%(figPath, name), bbox_inches='tight', pad_inches=0)
	plt.close()
	
def vtk_show(renderer, width=400, height=300, name='vtk'):
	renderWindow = vtk.vtkRenderWindow()
	renderWindow.SetOffScreenRendering(1)
	renderWindow.AddRenderer(renderer)
	renderWindow.SetSize(width, height)
	renderWindow.Render()
	 
	windowToImageFilter = vtk.vtkWindowToImageFilter()
	windowToImageFilter.SetInput(renderWindow)
	windowToImageFilter.Update()
	 
	writer = vtk.vtkPNGWriter()
	writer.SetWriteToMemory(1)
	writer.SetInputConnection(windowToImageFilter.GetOutputPort())
	writer.Write()
	
def save_fig(vtk_rw, name='MC'):
	vtk_win_im = vtk.vtkWindowToImageFilter()
	vtk_win_im.SetInput(vtk_rw)
	vtk_win_im.Update()

	vtk_image = vtk_win_im.GetOutput()

	width, height, _ = vtk_image.GetDimensions()
	vtk_array = vtk_image.GetPointData().GetScalars()
	components = vtk_array.GetNumberOfComponents()

	arr = vtk_to_numpy(vtk_array).reshape(height, width, components)
	arr = np.rot90(np.rot90(arr, -1), -1)

	plt.imshow(arr)
	plt.axis('off')
	plt.savefig('%s/%s.pdf'%(figPath, name), bbox_inches='tight', pad_inches=0)
	plt.close()

# DICOM Input
# Load and read-in the DICOM files
reader = vtk.vtkDICOMImageReader()
reader.SetDirectoryName(dicomPath)
reader.Update()

# Read in meta-data
# Load dimensions using `GetDataExtent`
_extent = reader.GetDataExtent()
ConstPixelDims = [_extent[1]-_extent[0]+1, _extent[3]-_extent[2]+1, _extent[5]-_extent[4]+1]

# Load spacing values
ConstPixelSpacing = reader.GetPixelSpacing()

ArrayDicom = vtkImageToNumPy(reader.GetOutput(), ConstPixelDims)
plotHeatmap(np.rot90(ArrayDicom[:, 256, :], -1), name="CT_Original")

# Use the `vtkImageThreshold` to clean all soft-tissue from the image data
threshold = vtk.vtkImageThreshold()
threshold.SetInputConnection(reader.GetOutputPort())
threshold.ThresholdByLower(400)  # remove all soft tissue
threshold.ReplaceInOn()
threshold.SetInValue(0)  # set all values below 400 to 0
threshold.ReplaceOutOn()
threshold.SetOutValue(1)  # set all values above 400 to 1
threshold.Update()

ArrayDicom = vtkImageToNumPy(threshold.GetOutput(), ConstPixelDims)
plotHeatmap(np.rot90(ArrayDicom[:, 256, :], -1), name="CT_Thresholded")

# time
dmc = vtk.vtkDiscreteMarchingCubes()
dmc.SetInputConnection(threshold.GetOutputPort())
dmc.GenerateValues(1, 1, 1)
dmc.Update()

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(dmc.GetOutputPort())

actor = vtk.vtkActor()
actor.SetMapper(mapper)

renderer = vtk.vtkRenderer()
renderer.AddActor(actor)
renderer.SetBackground(0, 0, 0)

camera = renderer.MakeCamera()
camera.SetPosition(-500.0, 245.5, 122.0)
camera.SetFocalPoint(301.0, 245.5, 122.0)
camera.SetViewAngle(30.0)
camera.SetRoll(-90.0)
renderer.SetActiveCamera(camera)
vtk_show(renderer, 480, 600, name='vtk_1')

camera = renderer.GetActiveCamera()
camera.SetPosition(301.0, 1045.0, 122.0)
camera.SetViewAngle(30.0)
camera.SetRoll(0.0)
renderer.SetActiveCamera(camera)
vtk_show(renderer, 480, 600, name='vtk_2')

# Save the extracted surface as an .stl file
writer = vtk.vtkSTLWriter()
writer.SetInputConnection(dmc.GetOutputPort())
writer.SetFileTypeToBinary()
writer.SetFileName("bones.stl")
writer.Write()

renderer_window = vtk.vtkRenderWindow()
renderer_window.SetWindowName("Bones")
renderer_window.AddRenderer(renderer) 
renderer_interactor = vtk.vtkRenderWindowInteractor()
renderer_interactor.SetRenderWindow(renderer_window)
renderer.SetBackground(0, 0, 0)
renderer_window.SetSize(480, 600)

renderer_interactor.Initialize()
save_fig(renderer_window)
renderer_window.Render()
renderer_interactor.Start()
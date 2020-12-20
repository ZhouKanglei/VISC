# This example reads a volume dataset and displays it via volume rendering(体绘制).

import vtk
import numpy as np
from vtk.util.misc import vtkGetDataRoot
from vtk.util.numpy_support import vtk_to_numpy

from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = 'STZhongsong'
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12   
plt.rcParams['axes.unicode_minus'] = False

dicomPath = "./vhm_head/"
figPath = 'E:/Documents/Study/VISC/Transfer function/figs'

def save_fig(vtk_rw, name='RC'):
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

# Create the renderer, the render window, and the interactor. The renderer
# draws into the render window, the interactor enables mouse- and
# keyboard-based interaction with the scene.
ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

# The following reader is used to read a series of 2D slices (images)
# that compose the volume. The slice dimensions are set, and the
# pixel spacing. The data Endianness must also be specified. The reader
# usese the FilePrefix in combination with the slice number to construct
# filenames using the format FilePrefix.%d. (In this case the FilePrefix
# is the root name of the file: quarter.)

# v16 = vtk.vtkVolume16Reader()
# v16.SetDataDimensions(64, 64)
# v16.SetImageRange(1, 93)
# v16.SetDataByteOrderToLittleEndian()
# v16.SetFilePrefix("D:/dicom_image/headsq/quarter")
# v16.SetDataSpacing(3.2, 3.2, 1.5)
v16 = vtk.vtkDICOMImageReader()
# v16.SetDirectoryName('D:/dicom_image/vtkDicomRender-master/sample')
v16.SetDirectoryName(dicomPath)

# The volume will be displayed by ray-cast alpha compositing.
# A ray-cast mapper is needed to do the ray-casting, and a
# compositing function is needed to do the compositing along the ray.
volumeMapper = vtk.vtkGPUVolumeRayCastMapper()
volumeMapper.SetInputConnection(v16.GetOutputPort())
volumeMapper.SetBlendModeToComposite()

# The color transfer function maps voxel intensities to colors.
# It is modality-specific, and often anatomy-specific as well.
# The goal is to one color for flesh (between 500 and 1000)
# and another color for bone (1150 and over).
volumeColor = vtk.vtkColorTransferFunction()
volumeColor.AddRGBPoint(0,    0.0, 0.0, 0.0)
volumeColor.AddRGBPoint(500,  1.0, 0.5, 0.3)
volumeColor.AddRGBPoint(1000, 1.0, 0.5, 0.3)
volumeColor.AddRGBPoint(1150, 1.0, 1.0, 0.9)

# The opacity transfer function is used to control the opacity
# of different tissue types.
volumeScalarOpacity = vtk.vtkPiecewiseFunction()
volumeScalarOpacity.AddPoint(0,    0.00)
volumeScalarOpacity.AddPoint(500,  0.15)
volumeScalarOpacity.AddPoint(1000, 0.15)
volumeScalarOpacity.AddPoint(1150, 0.85)

# The gradient opacity function is used to decrease the opacity
# in the "flat" regions of the volume while maintaining the opacity
# at the boundaries between tissue types.  The gradient is measured
# as the amount by which the intensity changes over unit distance.
# For most medical data, the unit distance is 1mm.
volumeGradientOpacity = vtk.vtkPiecewiseFunction()
volumeGradientOpacity.AddPoint(0,   0.0)
volumeGradientOpacity.AddPoint(90,  0.5)
volumeGradientOpacity.AddPoint(100, 1.0)

# The VolumeProperty attaches the color and opacity functions to the
# volume, and sets other volume properties.  The interpolation should
# be set to linear to do a high-quality rendering.  The ShadeOn option
# turns on directional lighting, which will usually enhance the
# appearance of the volume and make it look more "3D".  However,
# the quality of the shading depends on how accurately the gradient
# of the volume can be calculated, and for noisy data the gradient
# estimation will be very poor.  The impact of the shading can be
# decreased by increasing the Ambient coefficient while decreasing
# the Diffuse and Specular coefficient.  To increase the impact
# of shading, decrease the Ambient and increase the Diffuse and Specular.
volumeProperty = vtk.vtkVolumeProperty()
volumeProperty.SetColor(volumeColor)
volumeProperty.SetScalarOpacity(volumeScalarOpacity)
# volumeProperty.SetGradientOpacity(volumeGradientOpacity)
volumeProperty.SetInterpolationTypeToLinear()
volumeProperty.ShadeOn()
volumeProperty.SetAmbient(0.9)
volumeProperty.SetDiffuse(0.9)
volumeProperty.SetSpecular(0.9)

# The vtkVolume is a vtkProp3D (like a vtkActor) and controls the position
# and orientation of the volume in world coordinates.
volume = vtk.vtkVolume()
volume.SetMapper(volumeMapper)
volume.SetProperty(volumeProperty)

# Finally, add the volume to the renderer
ren.AddViewProp(volume)

# Set up an initial view of the volume.  The focal point will be the
# center of the volume, and the camera position will be 400mm to the
# patient's left (which is our right).
camera = ren.GetActiveCamera()
c = volume.GetCenter()
camera.SetFocalPoint(c[0], c[1], c[2])
camera.SetPosition(c[0] + 600, c[1], c[2])
camera.SetViewUp(0, 0, -1)
# camera.SetViewAngle(30.0)
# camera.SetRoll(90)

# Increase the size of the render window
renWin.SetSize(480, 600)

# Interact with the data.
iren.Initialize()
renWin.Render()
save_fig(renWin, 'RC')

iren.Start()


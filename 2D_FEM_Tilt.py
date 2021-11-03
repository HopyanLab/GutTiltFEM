#!/usr/bin/env /usr/bin/python3

import sys
import numpy as np
import pymesh
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import (
							FigureCanvasQTAgg as FigureCanvas,
							NavigationToolbar2QT as NavigationToolbar
							)
from matplotlib.figure import Figure
from matplotlib import collections as mc
from matplotlib.animation import FuncAnimation
from PyQt5.QtCore import Qt, QPoint, QRect, QSize
from PyQt5.QtGui import QDoubleValidator, QMouseEvent
from PyQt5.QtWidgets import (
							QApplication, QLabel, QWidget,
							QPushButton, QHBoxLayout, QVBoxLayout,
							QComboBox, QCheckBox, QSlider, QProgressBar,
							QFormLayout, QLineEdit, QTabWidget,
							QSizePolicy, QFileDialog, QMessageBox
							)
from numbers import Number
import pickle
import warnings
warnings.filterwarnings('ignore')

################################################################################

#def dNdx (nodes, faces):
#	N_e = faces.shape[0]
#	edges = np.vstack([faces[:,[1,2]],
#					   faces[:,[2,0]],
#					   faces[:,[0,1]]])
#	vects = nodes[edges[:,1]] - nodes[edges[:,0]]
#	norms = np.flip(vects,axis=-1)*np.array([1,-1])
#	areas = np.cross(vects[:N_e,:], vects[N_e:2*N_e,:])/2
#	print(np.any(areas < 0))
#	dN = np.zeros((N_e, 2, 3))
#	dN[:,:,0] = norms[0:N_e,:]/areas[:,np.newaxis]/2
#	dN[:,:,1] = norms[N_e:2*N_e,:]/areas[:,np.newaxis]/2
#	dN[:,:,2] = norms[2*N_e:3*N_e,:]/areas[:,np.newaxis]/2
#	return dN, areas

#def B_matrix (nodes, faces):
#	N_e = faces.shape[0]
#	N = nodes.shape[0]
#	dN, areas = dNdx(nodes, faces)
#	B = np.zeros((N_e, 3, 6))
#	B[:,0,0::2] = dN[:,0,:] # dN_i/dx
#	B[:,1,1::2] = dN[:,1,:] # dN_i/dy
#	B[:,2,0::2] = dN[:,1,:] # dN_i/dy
#	B[:,2,1::2] = dN[:,0,:] # dN_i/dx
#	return B, areas

################################################################################
# The above works but is slightly slower than the below simplified definitions.

def B_matrix (nodes, faces):
	v_12 = nodes[faces[:,1]] - nodes[faces[:,0]]
	v_23 = nodes[faces[:,2]] - nodes[faces[:,1]]
	v_31 = nodes[faces[:,0]] - nodes[faces[:,2]]
	areas = (np.cross(v_12, v_23))
	N_e = faces.shape[0]
	B = np.zeros((N_e, 3, 6))
	B[:,0,0] = v_23[:,1]/areas
	B[:,0,2] = v_31[:,1]/areas
	B[:,0,4] = v_12[:,1]/areas
	B[:,1,1] = -v_23[:,0]/areas
	B[:,1,3] = -v_31[:,0]/areas
	B[:,1,5] = -v_12[:,0]/areas
	B[:,2,0] = -v_23[:,0]/areas
	B[:,2,2] = -v_31[:,0]/areas
	B[:,2,4] = -v_12[:,0]/areas
	B[:,2,1] = v_23[:,1]/areas
	B[:,2,3] = v_31[:,1]/areas
	B[:,2,5] = v_12[:,1]/areas
	areas = areas/2
	return B, areas

################################################################################

def solve_FEM (nodes, faces, forces, boundary, E = 1.6, nu = 0.4):
	D = 1/(1-nu**2) * np.array([[1, nu, 0],
								[nu, 1, 0],
								[0, 0, (1-nu)/2]])
	N_e = faces.shape[0]
	N_n = nodes.shape[0]
	N_dof = 2*N_n
	reshaped_boundary = np.vstack([boundary,boundary]).T.reshape(N_dof)
	forces = forces.reshape(N_dof)
	forces = np.append(forces,
				np.zeros(np.count_nonzero(reshaped_boundary))) # Dirichlet BC
	B, areas = B_matrix(nodes, faces)
	K = np.zeros((N_dof, N_dof))
	if isinstance(E, Number):
		E = E * np.ones(faces.shape[0])
	for i in range(N_e):
		Ke = areas[i]*E[i]*(B[i,:,:].T.dot(D).dot(B[i,:,:]))
		idx = np.array(faces[i]*2)[np.newaxis]
		K[idx.T, idx] += Ke[0::2,0::2]		# XX
		K[(idx+1).T, idx+1] += Ke[1::2,1::2]# YY
		K[(idx+1).T, idx] += Ke[1::2,0::2]	# YX
		K[idx.T, idx+1] += Ke[0::2,1::2]	# XY
	K = np.append(K, np.identity(N_dof)[reshaped_boundary,:],
					axis = 0) # Dirichlet BC
	u = np.linalg.lstsq(K,forces, # np.linalg.solve requires square matrix
						rcond=None)[0].reshape((N_n,2))
	return u

################################################################################

def smooth_k_step (x,k=2): # k should be >= 1
	return np.where(x > 0, np.where(x<1,
		0.5*(np.tanh(k*(2*x-1)/2/np.sqrt(x*(1-x)))+1), 1), 0)

################################################################################

class FEM_system:
	def __init__ (self, nodes, faces):
		self.nodes = nodes
		self.faces = faces
		self.E_0  = 1.6e1			# Young's modulus
		self.nu_0 = 0.4			# Poisson ratio
		self.time_index = 0
		self.final_time = 8.
		self.delta_time = 0.1
		self.history = np.zeros(
					(int(np.floor(self.final_time / self.delta_time)) + 1,
					 self.nodes.shape[0], 2))
		self.history[0] = self.nodes.copy()
		self.E = self.E_0*np.ones(self.faces.shape[0])
		self.initial_nodes = self.nodes.copy()
		self.edges = np.unique(np.sort(
			np.vstack([np.stack((self.faces[:,0], self.faces[:,1]), axis=1),
					   np.stack((self.faces[:,1], self.faces[:,2]), axis=1),
					   np.stack((self.faces[:,2], self.faces[:,0]), axis=1)]),
							axis=1), axis=0)
		self.node_types = np.ones(self.nodes.shape[0], dtype = int)
		self.node_types[self.nodes[:,1] < 0.1] = 0 # boundary
		mask = np.logical_and(np.logical_not(self.nodes[:,1] < 0.1),
							  np.logical_and(self.nodes[:,1] < 1.75,
											 self.nodes[:,0] < 0))
		self.node_types[mask] = 2 # left
		mask = np.logical_and(np.logical_not(self.nodes[:,1] < 0.1),
							  np.logical_and(self.nodes[:,1] < 1.75,
											 self.nodes[:,0] > 0))
		self.node_types[mask] = 3 # right
		self.left_faces = np.logical_and(np.logical_and(
							self.node_types[self.faces[:,0]] == 2,
							self.node_types[self.faces[:,1]] == 2),
							self.node_types[self.faces[:,2]] == 2)
		self.right_faces = np.logical_and(np.logical_and(
							self.node_types[self.faces[:,0]] == 3,
							self.node_types[self.faces[:,1]] == 3),
							self.node_types[self.faces[:,2]] == 3)
		self.areas = np.abs(np.cross(self.nodes[self.faces[:,1],:] - \
									 self.nodes[self.faces[:,0],:],
									 self.nodes[self.faces[:,2],:] - \
									 self.nodes[self.faces[:,0],:])) * 0.5
		self.initial_areas = self.areas.copy()
		self.target_areas = self.areas.copy()
		self.forces = np.zeros_like(self.nodes)
	
	def reset(self):
		self.nodes = self.initial_nodes
		self.areas = self.initial_areas
		self.target_areas = self.initial_areas
		self.forces = np.zeros_like(self.nodes)
		self.E = self.E_0*np.ones(self.faces.shape[0])
	
	def recalculate(self, time):
		later_change = smooth_k_step((time-4)/2)
		early_change = smooth_k_step(time/4)
		self.E[self.left_faces] = self.E_0*(1 + 0.3*later_change)
		self.E[self.right_faces] = self.E_0*(1 - 0.2*early_change)
		self.target_areas[self.left_faces] = self.initial_areas[
													self.left_faces] * \
										(1 - 0.2*later_change)
		self.target_areas[self.right_faces] = self.initial_areas[
													self.right_faces] * \
										(1 + 0.3*early_change)
		self.areas = np.abs(np.cross(self.nodes[self.faces[:,1],:] - \
									 self.nodes[self.faces[:,0],:],
									 self.nodes[self.faces[:,2],:] - \
									 self.nodes[self.faces[:,0],:])) * 0.5
		self.forces = np.zeros_like(self.nodes)
		force_multiplier = np.log(self.target_areas / self.areas)
		vectors = self.nodes[self.faces[:,2],:] - \
				  self.nodes[self.faces[:,1],:]
		vectors = np.flip(vectors,axis=-1)*np.array([-1,1]) * \
						force_multiplier[:,np.newaxis]
		self.forces[:,0] += np.bincount(self.faces[:,0], vectors[:,0],
									minlength = self.nodes.shape[0])
		self.forces[:,1] += np.bincount(self.faces[:,0], vectors[:,1],
									minlength = self.nodes.shape[0])
		vectors = self.nodes[self.faces[:,0],:] - \
				  self.nodes[self.faces[:,2],:]
		vectors = np.flip(vectors,axis=-1)*np.array([-1,1]) * \
						force_multiplier[:,np.newaxis]
		self.forces[:,0] += np.bincount(self.faces[:,1], vectors[:,0],
									minlength = self.nodes.shape[0])
		self.forces[:,1] += np.bincount(self.faces[:,1], vectors[:,1],
									minlength = self.nodes.shape[0])
		vectors = self.nodes[self.faces[:,1],:] - \
				  self.nodes[self.faces[:,0],:]
		vectors = np.flip(vectors,axis=-1)*np.array([-1,1]) * \
						force_multiplier[:,np.newaxis]
		self.forces[:,0] += np.bincount(self.faces[:,2], vectors[:,0],
									minlength = self.nodes.shape[0])
		self.forces[:,1] += np.bincount(self.faces[:,2], vectors[:,1],
									minlength = self.nodes.shape[0])
		self.forces[self.node_types == 0, :] *= 0 # no forces on boundary
	
	def take_step(self):
		self.time_index += 1
		self.recalculate(self.time_index * self.delta_time)
		u = solve_FEM(self.nodes, self.faces, self.forces,
					  self.node_types == 0, self.E, self.nu_0)
		u[self.node_types == 0, :] = 0 # make sure Dirichlet BC enforced
		self.nodes += u
		self.history[self.time_index] = self.nodes.copy()
	
	def run_sim(self):
		for index in np.arange(int(np.floor(self.final_time / self.delta_time))):
			self.take_step()
	
	def plot_system(self, ax, time = 0.):
		if time > self.final_time:
			time = self.final_time
		time_index = int(np.floor(time / self.delta_time))
		lines = self.history[time_index, self.edges]
		lc = mc.LineCollection(lines, colors = 'black')
		ax.add_collection(lc)
		ax.plot(self.history[time_index, self.node_types == 0,0],
				self.history[time_index, self.node_types == 0,1],
					linestyle = '', marker = 'o',
					markersize = 3, color = 'red')
		ax.plot(self.history[time_index, self.node_types == 1,0],
				self.history[time_index, self.node_types == 1,1],
					linestyle = '', marker = 'o',
					markersize = 3, color = 'green')
		ax.plot(self.history[time_index, self.node_types == 2,0],
				self.history[time_index, self.node_types == 2,1],
					linestyle = '', marker = 'o',
					markersize = 3, color = 'orange')
		ax.plot(self.history[time_index, self.node_types == 3,0],
				self.history[time_index, self.node_types == 3,1],
					linestyle = '', marker = 'o',
					markersize = 3, color = 'blue')
		ax.set_xlim([-3.0,3.0])
		ax.set_ylim([-0.2,4.2])
		ax.invert_yaxis()

################################################################################

class MPLCanvas(FigureCanvas):
	def __init__ (self, parent=None, width=8, height=8, dpi=100):
		self.fig = Figure(figsize=(width, height), dpi=dpi)
		self.ax = self.fig.add_subplot(111)
		FigureCanvas.__init__(self, self.fig)
		self.setParent(parent)
		FigureCanvas.setSizePolicy(self,
				QSizePolicy.Expanding,
				QSizePolicy.Expanding)
		FigureCanvas.updateGeometry(self)
		self.fig.tight_layout()

################################################################################

class Window (QWidget):
	def __init__ (self):
		super().__init__()
		self.title = 'Gut Tube 2D FEM'
		meshfile = '2d_tube_model_mesh.stl'
		mesh = pymesh.load_mesh(meshfile)
		nodes = mesh.nodes[:,:2].copy()
		faces = mesh.faces.copy()
		self.model = FEM_system(nodes, faces)
		self.model.run_sim()
		with open('model.pkl','wb') as pickle_file:
			pickle.dump(self.model, pickle_file)
#		with open('model.pkl','rb') as pickle_file:
#			self.model = pickle.load(pickle_file)
		self.time = 0.
		self.canvas = MPLCanvas()
		self.toolbar = NavigationToolbar(self.canvas, self)
		self.setup_GUI()
	
	def setup_GUI (self):
		self.setWindowTitle(self.title)
		main_layout = QVBoxLayout()
		main_layout.addWidget(self.canvas)
		main_layout.addWidget(self.toolbar)
		slider_layout = QHBoxLayout()
		self.slider = QSlider(Qt.Horizontal)
		self.slider.setMinimum(0)
		self.slider.setMaximum(int(np.floor(self.model.final_time / \
											self.model.delta_time)))
		self.slider.setValue(0)
		self.slider.setSingleStep(1)
		self.slider.valueChanged.connect(self.slider_select)
		slider_layout.addWidget(self.slider)
		slider_layout.addWidget(QLabel('t:'))
		self.textbox = QLineEdit()
		self.textbox.setMaxLength(4)
		self.textbox.setFixedWidth(40)
		self.textbox.setText(f'{self.time:.1f}')
		self.textbox.setValidator(QDoubleValidator())
		self.textbox.editingFinished.connect(self.textbox_select)
		slider_layout.addWidget(self.textbox)
		main_layout.addLayout(slider_layout)
		self.setLayout(main_layout)
		self.plot()
	
	def slider_select (self):
		self.time = self.slider.value() * self.model.delta_time
		self.textbox.setText(f'{self.time:.1f}')
		self.plot()
	
	def textbox_select (self):
		input_time = float(self.textbox.text())
		if input_time < 0.:
			input_time = 0.
		if input_time > self.model.final_time:
			input_time = self.model.final_time
		self.time = input_time
		self.textbox.setText(f'{self.time:.1f}')
		self.slider.setValue(int(np.floor(input_time / \
											self.model.delta_time)))
		self.plot()
	
	def plot (self, time = 0):
		for image in self.canvas.ax.images:
			if image.colorbar is not None:
				image.colorbar.remove()
		for collection in self.canvas.ax.collections:
			if collection.colorbar is not None:
				collection.colorbar.remove()
		self.canvas.ax.clear()
		self.model.plot_system(self.canvas.ax, self.time)
		self.canvas.draw()

################################################################################

if __name__ == "__main__":
	app = QApplication(sys.argv)
	window = Window()
	window.show()
	sys.exit(app.exec_())

################################################################################
# EOF

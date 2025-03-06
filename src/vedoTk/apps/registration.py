from typing import Union, Tuple, Optional
from sys import argv
from os import remove

from numpy import array, ndarray, save, load
from vedo import Points, Mesh, Plotter, Line
from PyQt5 import QtWidgets
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

try:
    import Sofa
    import Sofa.Gui
    import Sofa.Simulation
except ModuleNotFoundError:
    print("[WARNING] Module vedoTk.apps.registration requires SOFA bindings installed.")

from vedoTk.utils.mesh import SurfaceSelection


def register(source: Union[Points, Mesh], target: Union[Points, Mesh]):
    """
    Register a source 3D object to another target 3D object.

    :param source: Source 3D object.
    :param target: Target 3D object.
    """

    # Create the Qt application
    app = QtWidgets.QApplication(argv)
    win = _QtApp(source=source, target=target)
    app.aboutToQuit.connect(win.close)

    # Launch the Vedo displays
    win.show()

    # Launch the Qt application
    app.exec_()


class _QtApp(QtWidgets.QMainWindow):

    def __init__(self, source: Union[Points, Mesh], target: Union[Points, Mesh]):
        """
        Qt window with tabs for mesh vertices selection and for mesh registration.

        :param source: Source 3D object.
        :param target: Target 3D object.
        """

        super().__init__()

        # Window size and title
        self.setObjectName('Registration')
        screen_size = QtWidgets.QDesktopWidget().availableGeometry().size()
        self.resize(screen_size * 0.8)

        # Define the central widget and layout
        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)

        # Define the tabs architecture
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs)
        tab_1 = QtWidgets.QWidget()
        tab_1.layout = QtWidgets.QVBoxLayout()
        tab_1.setLayout(tab_1.layout)
        tab_2 = QtWidgets.QWidget()
        tab_2.layout = QtWidgets.QVBoxLayout()
        tab_2.setLayout(tab_2.layout)
        self.tabs.addTab(tab_1, 'Selection')
        self.tabs.addTab(tab_2, 'Registration')
        self.tabs.currentChanged.connect(self.__change_tab_callback)

        # TAB1: add the selection viewer
        self.vtk_widget_selection = QVTKRenderWindowInteractor(self)
        self.plt_selection = _Selection(source=source, target=target, qt_widget=self.vtk_widget_selection)
        tab_1.layout.addWidget(self.vtk_widget_selection)

        # TAB1: add the options
        selection_options = QtWidgets.QWidget()
        selection_options.layout = QtWidgets.QGridLayout()
        selection_options.setLayout(selection_options.layout)
        tab_1.layout.addWidget(selection_options)
        selection_options.layout.addWidget(QtWidgets.QWidget(), 0, 2, 1, 2)
        # Checkbox for camera rotation
        self.share_cam_button = QtWidgets.QCheckBox('Sync Camera Rotation')
        self.share_cam_button.clicked.connect(self.__share_cam_callback)
        selection_options.layout.addWidget(self.share_cam_button, 0, 0, 1, 1)
        # Checkbox for index number display
        self.display_idx_button = QtWidgets.QCheckBox('Display Marker Indices')
        self.display_idx_button.clicked.connect(self.__display_idx_callback)
        selection_options.layout.addWidget(self.display_idx_button, 1, 0, 1, 1)
        # Button to clear selection
        clear_button = QtWidgets.QPushButton('Clear Selection')
        clear_button.clicked.connect(self.plt_selection.clear_selection)
        selection_options.layout.addWidget(clear_button, 0, 1, 1, 1)
        # Button to load selection
        load_button = QtWidgets.QPushButton('Load Selection')
        load_button.clicked.connect(self.__load_callback)
        selection_options.layout.addWidget(load_button, 1, 1, 1, 1)
        # Button to save selection
        save_button = QtWidgets.QPushButton('Save Selection')
        save_button.clicked.connect(self.__save_callback)
        selection_options.layout.addWidget(save_button, 2, 1, 1, 1)

        # TAB2: add the registration viewer
        self.vtk_widget_registration = QVTKRenderWindowInteractor(self)
        self.plt_registration = _Registration(source=source, target=target, qt_widget=self.vtk_widget_registration)
        tab_2.layout.addWidget(self.vtk_widget_registration)

        # TAB2: add the options
        registration_options = QtWidgets.QWidget()
        registration_options.layout = QtWidgets.QGridLayout()
        registration_options.setLayout(registration_options.layout)
        tab_2.layout.addWidget(registration_options)
        label = QtWidgets.QLabel('Recommended options: \n'
                                 ' - step 1: run a rigid registration with markers only \n'
                                 ' - step 2: run a non-rigid registration with ICP only')
        registration_options.layout.addWidget(label, 0, 2, 2, 2)
        # Checkbox for rigid registration
        self.rigid_button = QtWidgets.QCheckBox('Rigid registration')
        self.rigid_button.setChecked(True)
        registration_options.layout.addWidget(self.rigid_button, 0, 0, 1, 1)
        # Checkbox for markers springs
        self.rsff_button = QtWidgets.QCheckBox('Use markers springs')
        self.rsff_button.setChecked(True)
        registration_options.layout.addWidget(self.rsff_button, 1, 0, 1, 1)
        # Checkbox for ICP force field
        self.icp_button = QtWidgets.QCheckBox('Use ICP')
        registration_options.layout.addWidget(self.icp_button, 2, 0, 1, 1)
        # Checkbox for SOFA GUI
        self.sofa_button = QtWidgets.QCheckBox('Launch in SOFA GUI')
        registration_options.layout.addWidget(self.sofa_button, 0, 1, 1, 1)
        # Button for registration
        self.register_button = QtWidgets.QPushButton('Register')
        self.register_button.setStyleSheet('padding: 10px;')
        self.register_button.clicked.connect(self.__register_callback)
        registration_options.layout.addWidget(self.register_button, 1, 1, 2, 1)

        # Launch the viewers
        self.plt_selection.show()
        self.plt_registration.show()

    def close(self):
        """
        Shutdown protocol.
        """

        # Close the vtk renderers
        self.vtk_widget_selection.close()
        self.vtk_widget_registration.close()

    def __load_callback(self):
        """
        Load selection button callback.
        """

        # Get the file to load
        file, _ = QtWidgets.QFileDialog.getOpenFileName(self, caption='Load Selection', filter='Numpy ( *.npy)')

        # Load source and target selections
        selections = load(file)
        self.plt_selection.set_selection_idx(source_idx=selections[:len(selections) // 2],
                                             target_idx=selections[len(selections) // 2:])

    def __save_callback(self):
        """
        Save selection button callback.
        """

        # Get source and target selections
        source_selection, target_selection = self.plt_selection.get_selection_idx()
        if len(source_selection) == 0:
            return
        if len(source_selection) != len(target_selection):
            self.__warning_marker_len()
            return

        # Get the file to save selection
        file, _ = QtWidgets.QFileDialog.getSaveFileName(self, caption='Save Selection', directory='selection.npy')
        if file != '':
            save(file, array(source_selection.tolist() + target_selection.tolist()))

    def __share_cam_callback(self):
        """
        Share camera button callback.
        """

        self.plt_selection.share_cam = self.share_cam_button.isChecked()

    def __display_idx_callback(self):
        """
        Display idx button callback.
        """

        self.plt_selection.display_idx = self.display_idx_button.isChecked()
        self.plt_selection.update_idx()

    def __register_callback(self):
        """
        Registration button callback.
        """

        self.plt_registration.register(rigid=self.rigid_button.isChecked(),
                                       springs=self.rsff_button.isChecked(),
                                       icp=self.icp_button.isChecked(),
                                       gui=self.sofa_button.isChecked(),
                                       register_button=self.register_button)

    def __change_tab_callback(self):
        """
        Tab change callback.
        """

        # Update the registration viewer
        if self.tabs.currentIndex() == 1:

            # Display the selected positions in source and target meshes
            valid = self.plt_registration.add_markers(*self.plt_selection.get_selection_idx())
            self.register_button.setEnabled(valid)
            if not valid:
                self.__warning_marker_len()

    def __warning_marker_len(self):

        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        msg.setWindowTitle('Warning')
        msg.setText('Mismatching number of markers in source and target meshes.')
        msg.exec_()


class _Selection(Plotter):

    def __init__(self, source: Union[Points, Mesh], target: Union[Points, Mesh], qt_widget: QVTKRenderWindowInteractor):
        """
        Viewer to display the source and target mesh and to select corresponding vertices.

        :param source: Source 3D object.
        :param target: Target 3D object.
        :param qt_widget: Render in a QtWidget using an QVTKRenderWindowInteractor.
        """

        super().__init__(N=2, interactive=True, sharecam=False, qt_widget=qt_widget)

        # Add the target mesh selection
        self.at(1)
        self.__target_selection = SurfaceSelection(mesh=target)
        self.__target_selection.init(plt=self)

        # Add the source mesh selection
        self.at(0)
        self.__source_selection = SurfaceSelection(mesh=source)
        self.__source_selection.init(plt=self)

        # Add callbacks
        self.add_callback(event_name='MouseMove', func=self.__callback_mouse_move)
        self.add_callback(event_name='LeftButtonPress', func=self.__callback_left_click)
        self.add_callback(event_name='RightButtonPress', func=self.__callback_right_click)
        self.add_callback(event_name='InteractionEvent', func=self.__callback_interactor)

        # Options variables
        self.share_cam = False
        self.display_idx = False
        self.label_idx = None

    def get_selection_idx(self) -> Tuple[ndarray, ndarray]:
        """
        Return the selected indices in the source and the target meshes.
        """

        return self.__source_selection.selected_idx, self.__target_selection.selected_idx

    def set_selection_idx(self, source_idx: ndarray, target_idx: ndarray) -> None:
        """
        Define the selected indices for the source and target meshes.
        """

        # Clear and update the source selection
        self.at(0)
        self.__source_selection.clear()
        self.__source_selection.add(idx=source_idx)
        self.__source_selection.update_selection(plt=self, point_size=15)

        # Clear and update the target selection
        self.at(1)
        self.__target_selection.clear()
        self.__target_selection.add(idx=target_idx)
        self.__target_selection.update_selection(plt=self, point_size=15)

        # Update the selections labels
        self.update_idx()

    def clear_selection(self) -> None:
        """
        Clear the source and target markers selections.
        """

        # Clear the source selection
        self.at(0)
        self.__source_selection.clear()
        self.__source_selection.update_selection(plt=self)

        # Clear and update the target selection
        self.at(1)
        self.__target_selection.clear()
        self.__target_selection.update_selection(plt=self)

        # Update the selection labels
        self.update_idx()

    def update_idx(self) -> None:
        """
        Update the labels for the makers indices.
        """

        # Remove previous labels
        if self.label_idx is not None:
            self.remove(self.label_idx[0], at=0)
            self.remove(self.label_idx[1], at=1)
            self.label_idx = None

        # Create new labels
        if self.display_idx:
            self.label_idx = [Points(self.__source_selection.selected_points_coord).labels2d('pointid'),
                              Points(self.__target_selection.selected_points_coord).labels2d('pointid')]
            self.add(self.label_idx[0], at=0)
            self.add(self.label_idx[1], at=1)

        # Update display
        self.render()

    def __callback_interactor(self, event):
        """
        InteractionEvent callback.
        """

        # Camera rotation option is activated
        if self.share_cam:

            # Get the active renderer
            cur = self.interactor.GetInteractorStyle().GetCurrentRenderer()

            # Apply the interactor rotation in every other renderers
            for r in self.renderers:
                if r != cur:
                    self.interactor.GetInteractorStyle().SetCurrentRenderer(r)
                    self.interactor.GetInteractorStyle().Rotate()

            # Set back the active renderer
            self.interactor.GetInteractorStyle().SetCurrentRenderer(cur)

    def __callback_mouse_move(self, event) -> None:
        """
        MouseMoveEvent callback.
        """

        # Get the current mesh (source or target)
        selection = self.__source_selection if event.at == 0 else self.__target_selection

        # Cursor on the mesh: color the hovered point
        if event.actor:
            try:
                cursor = selection.get_closest_points(pos=event.picked3d)
            except TypeError:
                cursor = array([])

        # Cursor out of the mesh: uncolor the hovered point
        else:
            cursor = array([])

        # Update colors
        selection.update_selection(plt=self, cursor_idx=cursor, point_size=15)
        self.render()

    def __callback_left_click(self, event) -> None:
        """
        LeftButtonPressEvent callback.
        """

        # Cursor on the mesh: add the point to selection
        if event.actor:

            # Get the current mesh (source or target)
            selection = self.__source_selection if event.at == 0 else self.__target_selection
            cursor = selection.get_closest_points(pos=event.picked3d)

            # Add the cursor to selection and update colors
            if len(cursor) > 0:
                selection.add(idx=cursor)
                selection.update_selection(plt=self, cursor_idx=cursor, point_size=15)
                self.update_idx()
                self.render()

    def __callback_right_click(self, event):
        """
        RightButtonPressEvent callback.
        """

        # Cursor on the mesh: remove the point from selection
        if event.actor:

            # Get the current mesh (source or target)
            selection = self.__source_selection if event.at == 0 else self.__target_selection
            cursor = selection.get_closest_points(pos=event.picked3d)

            # Remove the cursor from selection and update colors
            if len(cursor) > 0:
                selection.remove(idx=cursor)
                selection.update_selection(plt=self, cursor_idx=array([]), point_size=15)
                self.update_idx()
                self.render()


class _Registration(Plotter):

    def __init__(self, source: Union[Points, Mesh], target: Union[Points, Mesh], qt_widget: QVTKRenderWindowInteractor):
        """
        Viewer to display the source and target mesh and to select corresponding vertices.

        :param source: Source 3D object.
        :param target: Target 3D object.
        :param qt_widget: Render in a QtWidget using an QVTKRenderWindowInteractor.
        """

        super().__init__(interactive=True, qt_widget=qt_widget)

        # Create the source 3D object and the markers
        self.source = source.copy()
        self.source.wireframe(True).c('orange')
        self.source_markers = None
        self.source_markers_idx = None

        # Create the target 3D object and the markers
        self.target = target.copy()
        self.target.point_size(5).c('purple')
        self.target_markers = None
        self.target_markers_idx = None

        # Matching markers lines
        self.lines = []

        # Add 3D objects to the viewer
        self.add(self.source, self.target)

        # Registration scene
        self.sofa_scene: Optional[_SOFA_Registration] = None

    def add_markers(self, source_idx: ndarray, target_idx: ndarray) -> bool:
        """
        Display the selected markers in source and target meshes.

        :param source_idx: Selected 3D markers indices in the source mesh.
        :param target_idx: Selected 3D markers indices in the target mesh.
        """

        # Remove the previous markers and lines
        self.remove(self.source_markers, self.target_markers, *self.lines)
        self.lines = []

        # Do not add the markers if their number mismatches
        if source_idx.shape != target_idx.shape:
            return False

        # Add the new markers
        self.source_markers_idx, self.target_markers_idx = source_idx, target_idx
        self.source_markers = Points(self.source.vertices[source_idx], c='orange', r=15)
        self.target_markers = Points(self.target.vertices[target_idx], c='purple', r=15)
        self.lines = [Line(s_p, t_p) for s_p, t_p in zip(self.source_markers.vertices, self.target_markers.vertices)]
        self.add(self.source_markers, self.target_markers, *self.lines)
        self.render()
        return True

    def register(self, rigid: bool, springs: bool, icp: bool, gui: bool, register_button: QtWidgets.QPushButton):
        """
        Launch the SOFA scene for registration.

        :param rigid: If True, compute a rigid registration.
        :param springs: If True, add a ResShapeSpringsForceField between markers.
        :param icp: If True, add a ClosestPointRegistrationForceField between 3D objects.
        :param gui: If True, launch the scene in the SOFA GUI.
        :param register_button: The pushed Qt button (to disable/enable during registration).
        """

        # Save source and target meshes as obj (easier to load in SOFA)
        source_file, target_file = 'temp_source.obj', 'temp_target.obj'
        self.source.write(source_file)
        self.target.write(target_file)

        # Launch the registration
        self.sofa_scene = _SOFA_Registration(source_file=source_file, target_file=target_file,
                                             source_markers=self.source_markers_idx,
                                             target_markers=self.target_markers_idx,
                                             rigid=rigid, springs=springs, icp=icp)
        Sofa.Simulation.initRoot(self.sofa_scene.root)

        # Delete temp files
        remove(source_file)
        remove(target_file)

        # Launch the scene
        if gui:
            self.__launch_in_sofa()
        else:
            self.__launch_in_vedo(register_button=register_button)

    def __launch_in_sofa(self):
        """
        Launch the registration scene in the SOFA GUI.
        """

        # Launch the SOFA GUI
        Sofa.Gui.GUIManager.Init('Scene', 'qglviewer')
        Sofa.Gui.GUIManager.createGUI(self.sofa_scene.root, __file__)
        Sofa.Gui.GUIManager.SetDimension(800, 600)
        Sofa.Gui.GUIManager.MainLoop(self.sofa_scene.root)
        Sofa.Gui.GUIManager.closeGUI()

        # Update the viewer with registered models
        self.__update_view()

    def __launch_in_vedo(self, register_button: QtWidgets.QPushButton):
        """
        Launch the registration scene in batch with an updated rendering in the viewer.
        """

        def timer_callback(event):

            # Registration is still running
            if not self.sofa_scene.is_done():

                # Trigger a time step in the simulation
                Sofa.Simulation.animate(self.sofa_scene.root, self.sofa_scene.root.dt.value)

                # Update the rendering
                self.__update_view()

            # Registration is complete
            else:

                # Stop the timer callback
                self.timer_callback('destroy', self.tid)

                # Enable the registration button again
                register_button.setEnabled(True)

        # Disable the registration button while SOFA is running
        register_button.setEnabled(False)

        # Start the timer callback
        self.add_callback(event_name='timer', func=timer_callback)
        self.tid = self.timer_callback('create')

    def __update_view(self):
        """
        Update the rendering view with the simulation visual data.
        """

        # Update the source mesh
        self.source.vertices = self.sofa_scene.ogl.position.value.copy()
        self.source.compute_normals()

        # Update the source markers and the matching lines
        self.source_markers.vertices = self.sofa_scene.markers.position.value.copy()
        self.remove(*self.lines)
        self.lines = [Line(s_p, t_p) for s_p, t_p in zip(self.source_markers.vertices,
                                                         self.target_markers.vertices)]
        self.add(*self.lines)

        # Update the display
        self.render()


class _SOFA_Registration(Sofa.Core.Controller):

    def __init__(self, source_file: str, target_file: str, source_markers: ndarray, target_markers: ndarray,
                 rigid: bool, springs: bool, icp: bool, *args, **kwargs):
        """
        The SOFA simulation for 3D objects registration.

        :param source_file: Path to the source 3D object.
        :param target_file: Path to the target 3D object.
        :param source_markers: Indices of the source markers.
        :param target_markers: Indices of the target markers.
        :param rigid: If True, compute a rigid registration.
        :param springs: If True, add a ResShapeSpringsForceField between markers.
        :param icp: If True, add a ClosestPointRegistrationForceField between 3D objects.
        """

        super().__init__(name='PyController', *args, **kwargs)

        # Rigid registration option
        self.rigid = rigid

        # Root node
        self.root = Sofa.Core.Node('root')
        self.root.dt.value = 0.1
        self.root.gravity.value = [0, 0, 0]
        self.root.addObject(self)

        # Plugins and rendering style
        plugin_list = [
            'Sofa.Component.Collision.Geometry',
            'Sofa.Component.Engine.Generate',
            'Sofa.Component.Engine.Select',
            'Sofa.Component.IO.Mesh',
            'Sofa.Component.LinearSolver.Iterative',
            'Sofa.Component.Mapping.Linear',
            'Sofa.Component.Mass',
            'Sofa.Component.ODESolver.Backward',
            'Sofa.Component.SolidMechanics.FEM.Elastic',
            'Sofa.Component.SolidMechanics.Spring',
            'Sofa.Component.StateContainer',
            'Sofa.Component.Topology.Container.Constant',
            'Sofa.Component.Topology.Container.Dynamic',
            'Sofa.Component.Topology.Container.Grid',
            'Sofa.Component.Visual',
            'Sofa.GL.Component.Rendering3D',
            'MultiThreading',
            'Registration']
        self.root.addObject('RequiredPlugin', pluginName=plugin_list)
        self.root.addObject('VisualStyle', displayFlags='showCollisionModels showForceFields')
        self.root.addObject('DefaultAnimationLoop')

        # Target visual model
        self.root.addChild('target')
        self.root.target.addObject('MeshOBJLoader', name='TargetMesh', filename=target_file)
        self.root.target.addObject('OglModel', name='TargetOGL', src='@TargetMesh',
                                   primitiveType='POINTS', color=[0.5, 0.4, 1., 1.])

        # Target markers
        self.root.target.addChild('markers')
        target_markers_pos = self.root.target.getObject('TargetMesh').position.value[target_markers]
        self.root.target.markers.addObject('MechanicalObject', position=target_markers_pos)
        self.root.target.markers.addObject('SphereCollisionModel', radius=5.)

        # Source physical model with a sparse grid
        self.root.addChild('source')
        self.root.source.addObject('MeshOBJLoader', name='SourceMesh', filename=source_file)
        self.root.source.addObject('EulerImplicitSolver', rayleighStiffness=0., rayleighMass=0., vdamping=0.)
        self.root.source.addObject('CGLinearSolver', template='GraphScattered', iterations=1000, threshold=1e-6,
                                   tolerance=1e-6)
        self.root.source.addObject('SparseGridTopology', name='SparseGrid', src='@SourceMesh', n=[10, 10, 10])
        self.root.source.addObject('MechanicalObject', name='DOFs', src='@SparseGrid')
        self.root.source.addObject('ParallelHexahedronFEMForceField', name='FEM', topology='@SparseGrid',
                                   youngModulus=1e6, poissonRatio=0.4, method='large', printLog=False)
        self.root.source.addObject('DiagonalMass', totalMass=1)
        self.FEM = self.root.source.getObject('FEM')

        # Source visual model
        self.root.source.addChild('visual')
        self.root.source.visual.addObject('OglModel', name='Visual', src='@../SourceMesh', color=[1., 0.4, 0.5, 1])
        self.root.source.visual.addObject('BarycentricMapping', input='@../DOFs', output='@Visual')
        self.ogl = self.root.source.visual.getObject('Visual')

        # Source surface to apply external forces
        self.root.source.addChild('surface')
        self.root.source.surface.addObject('TriangleSetTopologyContainer', name='SurfaceTopo', src='@../SourceMesh')
        self.root.source.surface.addObject('MechanicalObject', name='SurfaceMO', src='@SurfaceTopo', showObject=False)
        self.root.source.surface.addObject('BarycentricMapping', input='@../DOFs', output='@SurfaceMO')
        # Add springs between the markers
        if springs:
            self.root.source.surface.addObject('RestShapeSpringsForceField', stiffness=1e3, angularStiffness=0,
                                          drawSpring=True, springColor=[1., 0., 0., 1.],
                                          points=source_markers, external_rest_shape='@../../target/markers')
            self.RSFF = self.root.source.surface.getObject('RestShapeSpringsForceField')
        # Add ICP force field between 3D objects
        if icp:
            self.root.source.surface.addObject('ClosestPointRegistrationForceField', template='Vec3d',
                                          sourceTriangles='@SurfaceTopo.triangles',
                                          sourceNormals='@../visual/Visual.normal',
                                          position='@../../target/TargetOGL.position',
                                          normals='@../../target/TargetOGL.normal',
                                          cacheSize=4, drawMode=1, drawColorMap=False, normalThreshold=0.7,
                                          theCloserTheStiffer=True, rejectBorders=True,
                                          stiffness=10, damping=0)
            self.ICPFF = self.root.source.surface.getObject('ClosestPointRegistrationForceField')

        # Source markers
        self.root.source.surface.addChild('markers')
        self.root.source.surface.markers.addObject('PointsFromIndices', name='PFI', position='@../SurfaceMO.position',
                                              indices=source_markers)
        self.root.source.surface.markers.addObject('MechanicalObject', name='MarkersOnLiverMesh',
                                              position='@PFI.indices_position')
        self.root.source.surface.markers.addObject('SphereCollisionModel', radius=5.0, color=[0., 1., 0., 1.])
        self.markers = self.root.source.surface.markers.getObject('MarkersOnLiverMesh')

    def idx_step(self):
        """
        Return the current time step index.
        """

        return int(self.root.time.value // self.root.dt.value)

    def is_done(self):
        """
        Return a flag for end of registration.
        """

        if self.rigid:
            return self.idx_step() > 30
        return self.idx_step() > 200

    def onAnimateEndEvent(self, *args, **kwargs):
        """
        Time step end event.
        """

        # For non-rigid registration, decrease the Young Modulus of the source object along the time steps
        if not self.rigid and self.FEM.youngModulus.value > 100 and self.idx_step() % 10 == 0:
            self.FEM.youngModulus.value = self.FEM.youngModulus.value / 2
            self.FEM.reinit()

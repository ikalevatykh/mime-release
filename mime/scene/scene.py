import os

os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"
os.environ["MESA_GLSL_VERSION_OVERRIDE"] = "330"

import pybullet as pb


class Scene(object):
    def __init__(self, load_egl=False):
        self._conn_mode = pb.DIRECT
        self._gui_resolution = (640, 480)
        self._connected = False
        self._simulation_step = 1.0 / 240.0
        self._observation_step = 1.0 / 10.0
        self._modders = []
        self._real_time = False
        self._load_egl = load_egl
        self._plugin = None
        self.client_id = None

    @property
    def simulation_step(self):
        """Physics simulation step (seconds)."""
        return self._simulation_step

    @simulation_step.setter
    def simulation_step(self, sec):
        self._simulation_step = sec

    @property
    def dt(self):
        """State observation step (seconds)."""
        return self._observation_step

    @dt.setter
    def dt(self, sec):
        self._observation_step = sec

    @property
    def gui_resolution(self):
        """Width and height (tuple) of the GUI window."""
        return self._gui_resolution

    @gui_resolution.setter
    def gui_resolution(self, wh):
        self._gui_resolution = wh

    def apply_modder(self, modder):
        """Apply scene modifications after creating."""
        self._modders.append(modder)

    def modder_reset(self, np_random):
        for modder in self._modders:
            modder.reset(self, self.np_random)

    def renders(self, gui=False, shared=False):
        """
        Show or not GUI window for your scene (hide by default).
        Args:
            gui - create standalone GUI window
            shared - connect to external server (use to connect VR server)
        """
        if shared:
            self._conn_mode = pb.SHARED_MEMORY
        elif gui:
            self._conn_mode = pb.GUI
        else:
            self._conn_mode = pb.DIRECT

    def reset(self, np_random):
        if not self._connected:
            self.connect()
            self.load(np_random)
        self._reset(np_random)
        pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 1)

    def connect(self):
        options = ""
        if pb.GUI == self._conn_mode:
            options = "--width={} --height={}".format(*self._gui_resolution)
        self.client_id = pb.connect(self._conn_mode, options=options)
        if self._load_egl:
            print("Loading egl plugin...")
            import pkgutil

            egl = pkgutil.get_loader("eglRenderer")
            self._plugin = pb.loadPlugin(
                egl.get_filename(), "_eglRendererPlugin", physicsClientId=self.client_id
            )
        if self.client_id < 0:
            raise Exception("Cannot connect to pybullet")
        if self._conn_mode == pb.GUI:
            pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
            pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 1)
            pb.configureDebugVisualizer(pb.COV_ENABLE_TINY_RENDERER, 0)
        pb.resetSimulation(physicsClientId=self.client_id)
        pb.setPhysicsEngineParameter(
            numSolverIterations=50,
            fixedTimeStep=self.simulation_step,
            physicsClientId=self.client_id,
        )
        pb.setGravity(0, 0, -9.8, physicsClientId=self.client_id)
        pb.configureDebugVisualizer(
            pb.COV_ENABLE_RENDERING, 0, physicsClientId=self.client_id
        )
        self._connected = True

    def step(self):
        for _ in range(int(self._observation_step // self._simulation_step)):
            self._step(self._simulation_step)
            if not self._real_time:
                pb.stepSimulation(physicsClientId=self.client_id)

    def close(self):
        try:
            if self._connected:
                if self._load_egl and self._plugin is not None:
                    pb.unloadPlugin(self._plugin, physicsClientId=self.client_id)
                pb.disconnect(physicsClientId=self.client_id)
        except pb.error:
            raise RuntimeError(pb.error)

    def load(self, np_random):
        self._load(np_random)
        for modder in self._modders:
            modder.load(self)

    def _load(self, np_random):
        """Called once when connecting to physics server
        Use to load objects in the scene
        """
        raise NotImplementedError

    def _reset(self, np_random):
        """Called once before start.
        Use to setup your scene."""
        raise NotImplementedError

    def _step(self, dt):
        """Called for each simulation step.
        Use to update controllers / get precise feedback."""
        raise NotImplementedError

import pybullet as pb

from . import link


class Joint(object):
    def __init__(self, body_id, joint_index, client_id):
        self._body_id = body_id
        self._joint_index = joint_index
        self.client_id = client_id

    @property
    def body_id(self):
        return self._body_id

    @property
    def joint_index(self):
        return self._joint_index

    @property
    def info(self):
        return JointInfo(self._body_id, self._joint_index, self.client_id)

    @property
    def state(self):
        return JointState(self._body_id, self._joint_index, self.client_id)

    @property
    def child_link(self):
        return link.Link(self._body_id, self._joint_index, self.client_id)

    def reset(self, position, velocity=0.):
        pb.resetJointState(self._body_id, self._joint_index,
                           position, velocity, physicsClientId=self.client_id)

    def control(self, **kwargs):
        pb.setJointMotorControl2(self._body_id, self._joint_index,
                                 physicsClientId=self.client_id, **kwargs)

    def enable_force_torque_sensor(self, enable=1):
        pb.enableJointForceTorqueSensor(self._body_id, self._joint_index, enable,
                                        physicsClientId=self.client_id)

    def __eq__(self, other):
        return self._body_id == other.body_id and \
               self._joint_index == other.joint_index

    def __repr__(self):
        return 'Joint({}:{})'.format(self._body_id, self._joint_index)


class JointArray(object):
    def __init__(self, body_id, joint_indices, client_id):
        self._body_id = body_id
        self._joint_indices = joint_indices
        self.client_id = client_id

    @property
    def body_id(self):
        return self._body_id

    @property
    def joint_indices(self):
        return self._joint_indices

    @property
    def info(self):
        return [JointInfo(self._body_id, i, self.client_id) for i in self._joint_indices]

    @property
    def state(self):
        return JointStateArray(self._body_id, self._joint_indices, self.client_id)

    def reset(self, positions):
        [pb.resetJointState(self._body_id, i, pos, physicsClientId=self.client_id)
         for i, pos in zip(self._joint_indices, positions)]

    def control(self, **kwargs):
        pb.setJointMotorControlArray(self._body_id, self._joint_indices,
                                     physicsClientId=self.client_id, **kwargs)

    def __getitem__(self, key):
        return Joint(self._body_id, self._joint_indices[key], self.client_id)

    def __len__(self):
        return len(self._joint_indices)

    def __iter__(self):
        for i in self._joint_indices:
            yield Joint(self._body_id, i, self.client_id)

    def __eq__(self, other):
        return self._body_id == other.body_id and \
               self._joint_indices == other.joint_indices

    def __repr__(self):
        return 'JointArray({}:{})'.format(self._body_id, self._joint_indices)


class JointInfo(object):
    def __init__(self, body_id, joint_index, client_id):
        self._body_id = body_id
        self.client_id = client_id
        self._info = pb.getJointInfo(body_id, joint_index, physicsClientId=self.client_id)

    @property
    def body_id(self):
        return self._body_id

    @property
    def index(self):
        return self._info[0]

    @property
    def joint_name(self):
        return self._info[1].decode('utf8')

    @property
    def type(self):
        return self._info[2]

    @property
    def is_fixed(self):
        return self.type == pb.JOINT_FIXED

    @property
    def is_revolute(self):
        return self.type == pb.JOINT_REVOLUTE

    @property
    def is_prismatic(self):
        return self.type == pb.JOINT_PRISMATIC

    @property
    def q_index(self):
        return self._info[3]

    @property
    def u_index(self):
        return self._info[4]

    @property
    def flags(self):
        return self._info[5]

    @property
    def damping(self):
        return self._info[6]

    @property
    def friction(self):
        return self._info[7]

    @property
    def lower_limit(self):
        return self._info[8]

    @property
    def upper_limit(self):
        return self._info[9]

    @property
    def limits(self):
        return self._info[8], self._info[9]

    @property
    def max_force(self):
        return self._info[10]

    @property
    def max_velocity(self):
        return self._info[11]

    @property
    def link_name(self):
        return self._info[12].decode('utf8')

    @property
    def axis(self):
        return self._info[13]

    @property
    def parent_frame_pos(self):
        return self._info[14]

    @property
    def parent_frame_orn(self):
        return self._info[15]

    @property
    def parent_index(self):
        return self._info[16]

    @property
    def parent(self):
        if self.parent_index == -1:
            return None
        return JointInfo(self.body_id, self.parent_index, self.client_id)

    @property
    def path(self):
        path = []
        i = self
        while i is not None:
            path.insert(0, i)
            i = i.parent
        return path


class JointState(object):
    def __init__(self, body_id, joint_index, client_id):
        self.client_id = client_id
        self._state = pb.getJointState(body_id, joint_index, physicsClientId=client_id)

    @property
    def position(self):
        return self._state[0]

    @property
    def velocity(self):
        return self._state[1]

    @property
    def reaction_forces(self):
        return self._state[2]

    @property
    def applied_joint_motor_torque(self):
        return self._state[3]


class JointStateArray(object):
    def __init__(self, body_id, joint_indices, client_id):
        self.client_id = client_id
        self._states = pb.getJointStates(body_id, joint_indices, physicsClientId=client_id)

    @property
    def positions(self):
        return [s[0] for s in self._states]

    @property
    def velocities(self):
        return [s[1] for s in self._states]

    @property
    def reaction_forces(self):
        return [s[2] for s in self._states]

    @property
    def applied_joint_motor_torques(self):
        return [s[3] for s in self._states]

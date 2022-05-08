import numpy as np


class HP:
    FREE = 0
    FIXED = 1
    RESTRICTED = 2
    CHOICE = 3

    def __init__(self, hp_type, value=None):
        self.value = value
        self.type = hp_type

    @classmethod
    def free(cls):
        return cls(HP.FREE)

    @classmethod
    def fixed(cls, value):
        return cls(HP.FIXED, value)

    @classmethod
    def restricted(cls, value=None):
        return cls(HP.RESTRICTED, value)

    def get_float(self, tuner, name, min_value, max_value, sampling=None, pn=None, pv=None):
        if self.type == HP.FREE or (self.type == HP.RESTRICTED and self.value is None):
            tmp = tuner.Float(name, min_value=min_value, max_value=max_value, sampling=sampling, parent_name=pn,
                              parent_values=pv)
        elif self.type == HP.FIXED or (self.type == HP.RESTRICTED and self.value is not None):
            tmp = tuner.Fixed(name, self.value)
        else:
            raise ValueError("HP type not found")
        return np.float32(tmp)

    def get_bool(self, tuner, name):
        if self.type == HP.FREE or self.type == HP.RESTRICTED:
            tmp = tuner.Boolean(name)
        elif self.type == HP.FIXED:
            tmp = tuner.Fixed(name, self.value)
        else:
            raise ValueError("HP type not found")
        return tmp

    def get_float_vec(self, tuner, name, length, min_value, max_value, step=None, sampling=None, pn=None, pv=None):
        if self.type == HP.FREE:
            tmp = [tuner.Float(name + ' ' + str(i), min_value=min_value, max_value=max_value, sampling=sampling,
                               parent_name=pn, parent_values=pv, step=step)
                   for i in range(length)]
        elif self.type == HP.FIXED:
            tmp = [tuner.Fixed(name + ' ' + str(i), self.value[i]) for i in range(length)]
        elif self.type == HP.RESTRICTED:
            tmp2 = tuner.Float(name + ' 0', min_value=min_value, max_value=max_value, sampling=sampling,
                               parent_name=pn,
                               parent_values=pv, step=step)
            tmp = [tmp2 for _ in range(length)]
        else:
            raise ValueError("HP type not found")
        return tmp


class HP_Manager:
    def __init__(self,
                 units=HP.free(),
                 norm_sub_reservoirs=HP.free(),
                 norm_inter_connectivity=HP.free(),
                 connectivity_subr=HP.free(),
                 connectivity_inter=HP.free(),
                 input_scaling=HP.free(),
                 bias_scaling=HP.free(),
                 leaky=HP.free(),
                 learning_rate=HP.free(),
                 gsr=HP.free(),
                 vsr=HP.free(),
                 use_norm2=False,
                 ):
        self.units = units
        self.connectivity_subr = connectivity_subr
        self.connectivity_inter = connectivity_inter
        self.norm_sub_reservoirs = norm_sub_reservoirs
        self.norm_inter_connectivity = norm_inter_connectivity
        self.use_norm2 = use_norm2
        self.input_scaling = input_scaling
        self.bias_scaling = bias_scaling
        self.leaky = leaky
        self.learning_rate = learning_rate
        self.gsr = gsr
        self.vsr = vsr

    def get_units(self, tuner, min_value=50, max_value=250, sampling=None, choise=[50, 75, 100]):
        if self.units.type == HP.FREE:
            tmp = tuner.Int('units', min_value=min_value, max_value=max_value, sampling=sampling)
        elif self.units.type == HP.RESTRICTED:
            tmp = tuner.Choise('units', choise, ordered=True)
        elif self.units.type == HP.FIXED:
            tmp = tuner.Fixed('units', self.units.value)
        else:
            raise ValueError("HP type not found")
        return tmp

    def get_connectivity_esn(self, tuner):
        if self.connectivity_subr.type == HP.FREE or self.connectivity_subr.type == HP.RESTRICTED:
            tmp = tuner.Float('connectivity 0', min_value=0.1, max_value=1., sampling="linear")
        elif self.connectivity_subr.type == HP.FIXED:
            tmp = tuner.Fixed('connectivity 0', self.connectivity_subr.value)
        else:
            raise ValueError("HP type not found")
        return tmp

    def get_connectivity_iresn(self, tuner, length):
        if self.connectivity_subr.type == HP.FREE:
            tmp = [tuner.Float('connectivity ' + str(i), min_value=0., max_value=1., sampling="linear") for i in
                   range(length)]
        elif self.connectivity_subr.type == HP.FIXED:
            tmp = [tuner.Fixed('connectivity 0', self.connectivity_subr.value) for _ in range(length)]
        elif self.connectivity_subr.type == HP.RESTRICTED:
            tmp2 = tuner.Float('connectivity 0', min_value=0., max_value=1., sampling="linear")
            tmp = [tmp2 for _ in range(length)]
        else:
            raise ValueError("HP type not found")
        return tmp

    def get_connectivity_iiresn(self, tuner, length):
        conn_subr = self.get_connectivity_iresn(tuner, length)
        if self.connectivity_inter.type == HP.FREE:
            conn_matrix = [[conn_subr[i] if i == j else
                            tuner.Float('connectivity ' + str(i) + '->' + str(j), min_value=0., max_value=1.,
                                        sampling="linear")
                            for i in range(length)]
                           for j in range(length)]
        elif self.connectivity_inter.type == HP.RESTRICTED:
            conn = tuner.Float('connectivity X->Y', min_value=0., max_value=1., sampling="linear")
            conn_matrix = [[conn_subr[i] if i == j else
                            conn for i in range(length)]
                           for j in range(length)]
        elif self.norm_inter_connectivity.type == HP.FIXED:
            conn_matrix = [[conn_subr[i] if i == j else
                            tuner.Fixed('connectivity X->Y', self.connectivity_inter.value)
                            for i in range(length)]
                           for j in range(length)]
        else:
            raise ValueError("HP type not found")
        return conn_matrix

    def get_minmax(self, tuner, length, min_value=0.01, max_value=1.5, sampling="linear"):
        if self.norm_inter_connectivity.type == HP.FREE:
            minmax_vec = [[0. if i == j else tuner.Float('minmax ' + str(i) + '->' + str(j), min_value=min_value,
                                                         max_value=max_value, sampling=sampling)
                           for i in range(length)]
                          for j in range(length)]
        elif self.norm_inter_connectivity.type == HP.RESTRICTED:
            minmax = tuner.Float('minmax', min_value=min_value, max_value=max_value, sampling=sampling)
            minmax_vec = [[0. if i == j else minmax
                           for i in range(length)]
                          for j in range(length)]
        elif self.norm_inter_connectivity.type == HP.FIXED:
            minmax = self.norm_inter_connectivity.value
            minmax_vec = [[0. if i == j else minmax
                           for i in range(length)]
                          for j in range(length)]
        else:
            raise ValueError("HP type not found")
        return minmax_vec

    def get_normalization_esn(self, tuner, min_value=0.01, max_value=1.5, sampling="linear"):
        return self.norm_sub_reservoirs.get_float(tuner, 'spectral radius', min_value=min_value,
                                                  max_value=max_value, sampling=sampling)

    def get_normalization_iresn(self, tuner, length, min_value=0.01, max_value=1.5, sampling="linear"):
        if self.norm_sub_reservoirs.type == HP.FREE:
            sr_vec = [
                tuner.Float('spectral radius ' + str(i), min_value=min_value, max_value=max_value,
                            sampling=sampling)
                for i
                in range(length)]
        elif self.norm_sub_reservoirs.type == HP.RESTRICTED:
            spectral_radius = tuner.Float('spectral radius 0', min_value=min_value, max_value=max_value,
                                          sampling=sampling)
            sr_vec = [spectral_radius for _ in range(length)]
        elif self.norm_sub_reservoirs.type == HP.FIXED:
            spectral_radius = self.norm_sub_reservoirs.value
            sr_vec = [spectral_radius for _ in range(length)]
        else:
            raise ValueError("HP type not found")
        return sr_vec

    def get_normalization_iiresn(self, tuner, length, min_value=0.01, max_value=1.5, sampling="linear"):
        sr_vec = self.get_normalization_iresn(tuner, length, min_value=min_value, max_value=max_value,
                                              sampling=sampling)
        if self.norm_inter_connectivity.type == HP.FREE:
            norm = [[sr_vec[i] if i == j else
                     tuner.Float('norm ' + str(i) + '->' + str(j), min_value=min_value, max_value=min_value,
                                 sampling=sampling)
                     for i in range(length)]
                    for j in range(length)]
        elif self.norm_inter_connectivity.type == HP.RESTRICTED:
            norm2 = tuner.Float('norm X->Y', min_value=min_value, max_value=min_value, sampling=sampling)
            norm = [[sr_vec[i] if i == j else
                     norm2 for i in range(length)]
                    for j in range(length)]
        elif self.norm_inter_connectivity.type == HP.FIXED:
            norm = [[sr_vec[i] if i == j else
                     tuner.Fixed('norm X->Y', self.norm_inter_connectivity.value)
                     for i in range(length)]
                    for j in range(length)]
        else:
            raise ValueError("HP type not found")
        return norm

    def get_input_scaling(self, tuner, length, min_value=0.01, max_value=1.5, sampling="linear"):
        if length != 1:
            return self.input_scaling.get_float_vec(tuner, 'input scaling', length, min_value=min_value,
                                                    max_value=max_value, sampling=sampling)
        else:
            return self.input_scaling.get_float(tuner, 'input scaling', min_value=min_value, max_value=max_value,
                                                sampling=sampling)

    def get_bias_scaling(self, tuner, length, min_value=0.01, max_value=1.5, sampling="linear"):
        if length != 1:
            return self.bias_scaling.get_float_vec(tuner, 'bias scaling', length, min_value=min_value,
                                                   max_value=max_value, sampling=sampling)
        else:
            return self.bias_scaling.get_float(tuner, 'bias scaling', min_value=min_value, max_value=max_value,
                                               sampling=sampling)

    def get_gsr(self, tuner, min_value=0.01, max_value=1.5, sampling="linear"):
        use_gsr = self.gsr.get_bool(tuner, 'use G.S.R.')
        if use_gsr:
            return tuner.Float('G.S.R.', min_value=min_value, max_value=max_value, sampling=sampling,
                               parent_name='use G.S.R.', parent_values=True)
        else:
            return None

    def get_vsr(self, tuner, sub_reservoirs):
        use_vsr = self.vsr.get_bool(tuner, 'use V.S.R.')
        if use_vsr:
            partitions = [tuner.Float('partition ' + str(i), min_value=0.1, max_value=1.0, sampling="linear") for i in
                          range(sub_reservoirs)]
            total = sum(partitions)
            # Normalize the partition vector now sum(partitions) == 1.
            partitions = list(map(lambda _x: 0 if total == 0 else _x / total, partitions))
            return partitions
        else:
            return None

    def get_leaky(self, tuner, min_value=0.0, max_value=1., sampling="linear"):
        return self.leaky.get_float(tuner, 'leaky', min_value=min_value, max_value=max_value, sampling=sampling)

    def get_learning_rate(self, tuner, min_value=1e-5, max_value=1e-1, sampling='log'):
        return self.leaky.get_float(tuner, 'learning rate', min_value=min_value, max_value=max_value, sampling=sampling)

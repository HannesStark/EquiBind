from torch.optim.lr_scheduler import *
import numpy as np


class WarmUpWrapper:
    "Optim wrapper that implements lr."

    def __init__(self, optimizer, wrapped_scheduler, warmup_steps, interpolation='linear',
                 **kwargs):
        '''

        :param optimizer:
        :param wrapped_scheduler:
        :param warmup_steps: is a list containing how many warmup steps should be done for each param group before updating all parameters
        :param interpolation:
        :param kwargs:
        '''
        self.optim = optimizer
        self._step = 0
        self.interpolation = interpolation
        self.warmup_steps = np.array(warmup_steps)
        self.total_warmup_steps = self.warmup_steps.sum()
        self.wrapped_scheduler = globals()[wrapped_scheduler](self.optim, **kwargs)
        self.start_lrs = []
        self.warmup_phase = 0
        for p in self.optim.param_groups:
            self.start_lrs.append(p['lr'])
            p['lr'] = 0

    def step(self, metrics=None):
        "Update parameters and lr"
        if self._step < self.total_warmup_steps:
            warmup_phase = 0
            for steps in self.warmup_steps.cumsum():
                if self._step >= steps:
                    warmup_phase += 1
            for i, p in enumerate(self.optim.param_groups):
                # update all parameters if there is only one entry specified for the warmup steps otherwise only update the ones corresponding to the current warmup phase
                if i <= warmup_phase or len(self.warmup_steps) == 1:
                    # interpolate between 0 and the final starting learning rate
                    interpolation_value = self._step - ([0] + list(self.warmup_steps.cumsum()))[warmup_phase] +1
                    if self.warmup_steps[warmup_phase] == 0:
                        p['lr'] = self.start_lrs[i]
                    else:
                        if self.interpolation == 'linear':
                            p['lr'] = self.start_lrs[i] * (interpolation_value / self.warmup_steps[warmup_phase])
                        elif self.interpolation == 'cosine':
                            p['lr'] = self.start_lrs[i] * (
                                    (-np.cos((np.pi) * (interpolation_value / self.warmup_steps[warmup_phase])) + 1) * 0.5)
                        else:
                            raise ValueError('interpolation not implemented:', self.interpolation)

        else:
            if metrics != None:
                self.wrapped_scheduler.step(metrics=metrics)
            else:
                self.wrapped_scheduler.step()
        self._step += 1

    def state_dict(self):
        """Returns the state of the warmup_steps scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optim.
        """
        state_dict = {key: value for key, value in self.__dict__.items() if key != 'optim'}
        state_dict['wrapped_scheduler'] = self.wrapped_scheduler.state_dict()  # overwrite with the state dict
        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the warmup_steps scheduler's state.
        Arguments:
            state_dict (dict): warmup_steps scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        wrapped_scheduler_state_dict = state_dict['wrapped_scheduler']
        del state_dict['wrapped_scheduler']
        self.wrapped_scheduler.load_state_dict(wrapped_scheduler_state_dict)
        self.__dict__.update(state_dict)

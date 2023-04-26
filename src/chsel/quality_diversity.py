import abc
from typing import Optional
import logging

import cma
import numpy as np

import torch
from arm_pytorch_utilities.tensor_utils import ensure_tensor
from chsel.conversion import RT_to_continuous_representation, continuous_representation_to_RT

from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter, GradientArborescenceEmitter
from ribs.schedulers import Scheduler
from chsel.costs import RegistrationCost
from chsel.types import SimilarityTransform, ICPSolution
from chsel import registration_util

logger = logging.getLogger(__name__)


class QDOptimization:
    def __init__(self, registration_cost: RegistrationCost,
                 model_points_world_frame: torch.tensor,
                 init_transform: Optional[SimilarityTransform] = None,
                 sigma=0.1,
                 num_emitters=5,
                 save_loss_plot=False,
                 savedir=registration_util.ROOT_DIR,
                 **kwargs):
        """

        :param registration_cost: some implementation of Equation 6
        :param model_points_world_frame: not actually used, but useful for tracking and debugging registration process
        can be points with known SDF value = 0 (surface points) in the world frame
        :param init_transform: transform from which to start the estimation
        :param sigma: QD parameter for specifying degree of exploration
        :param num_emitters: number of points to consider simultaneously, for example if the search space is large
        :param save_loss_plot: whether to plot losses and save them
        :param savedir: where to save the plotted losses
        :param kwargs: kwargs forwarded to creating the scheduler
        """
        self.registration_cost = registration_cost
        self.X = model_points_world_frame
        self.Xt = self.X
        self.B = self.Xt.shape[0]

        self.init_transform = init_transform
        self.sigma = sigma
        self.save_loss_plot = save_loss_plot
        self.savedir = savedir

        self.device = self.Xt.device
        self.dtype = self.Xt.dtype

        Xt, R, T, s = registration_util.apply_init_transform(self.Xt, self.init_transform)
        x = self.get_numpy_x(R, T)
        self.num_emitters = num_emitters
        self.scheduler = self.create_scheduler(x, **kwargs)

    def run(self):
        Xt, R, T, s = registration_util.apply_init_transform(self.Xt, self.init_transform)

        # initialize the transformation history
        t_history = []
        losses = []

        while not self.is_done():
            cost = self.step()

            losses.append(cost)

        R, T, rmse = self.process_final_results(None, losses)

        return ICPSolution(True, rmse, Xt, SimilarityTransform(R, T, s), t_history)

    @abc.abstractmethod
    def add_solutions(self, solutions):
        pass

    @abc.abstractmethod
    def create_scheduler(self, x0, *args, **kwargs):
        pass

    @abc.abstractmethod
    def step(self):
        pass

    @abc.abstractmethod
    def process_final_results(self, s, losses):
        pass

    @abc.abstractmethod
    def is_done(self):
        return False

    @abc.abstractmethod
    def get_all_elite_solutions(self):
        return None

    @staticmethod
    def get_numpy_x(R, T):
        return RT_to_continuous_representation(R, T).cpu().numpy()

    def get_torch_RT(self, x):
        return continuous_representation_to_RT(x, self.device, self.dtype)


class CMAES(QDOptimization):
    def create_scheduler(self, x, *args, **kwargs):
        x0 = x[0]
        options = {"popsize": self.B, "seed": np.random.randint(0, 10000), "tolfun": 1e-5, "tolfunhist": 1e-6}
        options.update(kwargs)
        es = cma.CMAEvolutionStrategy(x0=x0, sigma0=self.sigma, inopts=options)
        return es

    def is_done(self):
        return self.scheduler.stop()

    def step(self):
        solutions = self.scheduler.ask()
        # convert back to R, T, s
        R, T = self.get_torch_RT(np.stack(solutions))
        cost = self.registration_cost(R, T, None)
        self.scheduler.tell(solutions, cost.cpu().numpy())
        return cost

    def add_solutions(self, solutions):
        pass

    def get_all_elite_solutions(self):
        return None

    def process_final_results(self, s, losses):
        # convert ES back to R, T
        solutions = self.scheduler.ask()
        R, T = self.get_torch_RT(np.stack(solutions))
        rmse = self.registration_cost(R, T, s)

        if self.save_loss_plot:
            registration_util.plot_poke_losses(losses, savedir=self.savedir)

        return R, T, rmse


# try measure function on the rotation dimensions
def rot_measure(measure_dim, offset=0):
    # ensure the measure is only over the rotation dimensions
    assert measure_dim + offset <= 6

    def fn(x):
        return x[..., offset:measure_dim]

    def grad(x):
        grad = np.zeros((measure_dim, x.shape[-1]))
        grad[:, offset:measure_dim] = np.eye(measure_dim)
        grad = np.tile(grad, (x.shape[0], 1, 1))
        return grad

    return fn, grad


def position_measure(measure_dim):
    def fn(x):
        return x[..., 6:6 + measure_dim]

    def grad(x):
        grad = np.zeros((measure_dim, x.shape[-1]))
        grad[:, 6:6 + measure_dim] = np.eye(measure_dim)
        grad = np.tile(grad, (x.shape[0], 1, 1))
        return grad

    return fn, grad


class CMAME(QDOptimization):
    def __init__(self, *args, bins=20, iterations=100,
                 # require an explicit range
                 ranges=None,
                 outlier_ratio=5.0,  # reject solutions that are this many times worse than the best
                 qd_score_offset=-100,  # useful for tracking the archive QD score as monotonically increasing
                 measure_dim=2,  # how many dimensions of translation to use, in the order of XYZ
                 # custom measure function, overrides measure_dim
                 measure_fn=None, measure_grad=None,
                 **kwargs):
        self.measure_dim = measure_dim
        if "sigma" not in kwargs:
            kwargs["sigma"] = 1.0
        if isinstance(bins, (float, int)):
            self.bins = [bins for _ in range(self.measure_dim)]
        else:
            assert len(bins) == self.measure_dim
            self.bins = bins
        self.iterations = iterations
        self.ranges = ranges
        self.qd_score_offset = qd_score_offset
        self.outlier_ratio = outlier_ratio

        self.archive = None
        self.i = 0
        self.qd_scores = []

        self._measure = measure_fn
        self._measure_grad = measure_grad
        if self._measure is None:
            self._measure, self._measure_grad = position_measure(self.measure_dim)

        super(CMAME, self).__init__(*args, **kwargs)

    def _create_ranges(self):
        if self.ranges is None:
            raise RuntimeError("An explicit archive range must be specified")

    def create_scheduler(self, x, *args, **kwargs):
        self._create_ranges()
        self.archive = GridArchive(solution_dim=x.shape[1], dims=self.bins, ranges=self.ranges[:self.measure_dim],
                                   seed=np.random.randint(0, 10000), qd_score_offset=self.qd_score_offset)
        emitters = [
            EvolutionStrategyEmitter(self.archive, x0=x[i], sigma0=self.sigma, batch_size=self.B,
                                     seed=np.random.randint(0, 10000)) for i in
            range(self.num_emitters)
        ]
        scheduler = Scheduler(self.archive, emitters)
        return scheduler

    def is_done(self):
        return self.i >= self.iterations

    def step(self):
        self.i += 1
        solutions = self.scheduler.ask()
        # evaluate the models and record the objective and behavior
        # note that objective is -cost
        R, T = self.get_torch_RT(np.stack(solutions))
        cost = self.registration_cost(R, T, None)
        bcs = self._measure(solutions)
        self.scheduler.tell(-cost.cpu().numpy(), bcs)
        qd = self.archive.stats.norm_qd_score
        self.qd_scores.append(qd)
        logger.debug("step %d norm QD score: %f", self.i, qd)
        return cost

    def _add_solutions(self, solutions):
        R, T = self.get_torch_RT(np.stack(solutions))
        rmse = self.registration_cost(R, T, None)
        self.archive.add(solutions, -rmse.cpu().numpy(), self._measure(solutions))

    def add_solutions(self, solutions):
        if solutions is None:
            return
        assert isinstance(solutions, np.ndarray)
        SOLUTION_CHUNK = 300
        for i in range(0, solutions.shape[0], SOLUTION_CHUNK):
            self._add_solutions(solutions[i:i + SOLUTION_CHUNK])

    def get_all_elite_solutions(self):
        df = self.archive.as_pandas()
        solutions = df.solution_batch()
        return solutions

    def process_final_results(self, s, losses):
        df = self.archive.as_pandas()
        objectives = df.objective_batch()
        solutions = df.solution_batch()

        cost = -objectives
        # filter out all solutions that are more than outlier_ratio times worse than the best
        lowest_cost = np.min(cost)
        inlier_mask = cost < lowest_cost * self.outlier_ratio
        solutions = solutions[inlier_mask]
        cost = cost[inlier_mask]
        if len(solutions) > self.B:
            order = np.argpartition(cost, self.B)
            solutions = solutions[order[:self.B]]
        # if there are fewer than B solutions, randomly sample the remaining ones from existing solutions
        elif len(solutions) < self.B:
            # sample from existing solutions with replacement by index
            resampled_indices = np.random.choice(np.arange(len(solutions)), self.B - len(solutions))
            solutions = np.concatenate([solutions, solutions[resampled_indices]], axis=0)
        # convert back to R, T
        R, T = self.get_torch_RT(solutions)
        rmse = self.registration_cost(R, T, s)

        if self.save_loss_plot:
            registration_util.plot_poke_losses(losses, savedir=self.savedir)
            qd_scores = [torch.tensor(v).view(1) for v in self.qd_scores]
            registration_util.plot_poke_losses(qd_scores, savedir=self.savedir, loss_name='qd_score', logy=False,
                                               ylabel='norm qd score')
            registration_util.plot_qd_archive(self.archive, savedir=self.savedir)

        return R, T, rmse


class CMAMEGA(CMAME):
    def __init__(self, *args, lr=0.05, **kwargs):
        self.lr = lr
        super(CMAMEGA, self).__init__(*args, **kwargs)

    def create_scheduler(self, x, *args, **kwargs):
        self._create_ranges()
        self.archive = GridArchive(solution_dim=x.shape[1], dims=self.bins, seed=np.random.randint(0, 10000),
                                   ranges=self.ranges[:self.measure_dim], qd_score_offset=self.qd_score_offset)
        emitters = []
        # emitters += [
        #     EvolutionStrategyEmitter(self.archive, x0=x[i], sigma0=self.sigma, batch_size=self.B) for i in
        #     range(self.num_emitters)
        # ]
        # rb = 3
        # rot_bounds = np.array([[-rb, rb] for _ in range(6)])
        # bounds = np.concatenate((self.ranges, rot_bounds))
        emitters += [
            GradientArborescenceEmitter(self.archive, x0=x[i], sigma0=self.sigma, lr=self.lr, grad_opt="adam",
                                        selection_rule="filter", bounds=None, batch_size=self.B - 1,
                                        seed=np.random.randint(0, 10000)) for i in
            range(self.num_emitters)
        ]
        scheduler = Scheduler(self.archive, emitters)
        return scheduler

    def _f(self, x):
        R, T = self.get_torch_RT(x)
        return self.registration_cost(R, T, None)

    def step(self):
        solutions = self.scheduler.ask_dqd()
        bcs = self._measure(solutions)
        # evaluate the models and record the objective and behavior
        # note that objective is -cost
        # get objective gradient and also the behavior gradient
        x = ensure_tensor(self.device, self.dtype, solutions)
        x.requires_grad = True
        cost = self._f(x)
        cost.sum().backward()
        objective_grad = -(x.grad.cpu().numpy())
        objective = -cost.detach().cpu().numpy()
        objective_grad = objective_grad.reshape(x.shape[0], 1, -1)
        measure_grad = self._measure_grad(x)

        jacobian = np.concatenate((objective_grad, measure_grad), axis=1)
        self.scheduler.tell_dqd(objective, bcs, jacobian)

        return super(CMAMEGA, self).step()

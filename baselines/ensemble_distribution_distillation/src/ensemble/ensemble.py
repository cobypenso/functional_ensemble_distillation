"""Ensemble"""
import logging
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.optim as torch_optim
import src.metrics as metrics
import src.utils as utils


class Ensemble():
    """Ensemble base class
    The ensemble member needs to track the size
    of the output of the ensemble
    This can be automatically inferred but it would look ugly
    and this now works as a sanity check as well

    Instance variables:
        output_size (int): Represents the actual output size
            i.e. the number of predicted parameters.
            e.g. if we model D-dimensional target with a Gaussian
            with mean and diagonal covariance, the output size would be 2D.
    """
    def __init__(self, output_size, device=torch.device("cpu")):
        self.members = list()
        self._log = logging.getLogger(self.__class__.__name__)
        self.output_size = output_size
        self.device = device
        self.size = 0

    def __len__(self):
        return self.size

    def add_member(self, new_member):
        if issubclass(type(new_member), EnsembleMember) or True:
            self._log.warning("Is subclass check disabled")
            self._log.info("Adding {} to ensemble".format(type(new_member)))
            self.members.append(new_member)
            self.size += 1
        else:
            err_str = "Ensemble member must be an EnsembleMember subclass"
            self._log.error(err_str)
            raise ValueError(err_str)

    def add_multiple(self, number_of, constructor):
        for _ in range(number_of):
            self.add_member(constructor())

    def train(self, train_loader, num_epochs, validation_loader=None):
        """Multithreaded?"""
        self._log.info("Training ensemble")
        for ind, member in enumerate(self.members):
            self._log.info("Training member {}/{}".format(ind + 1, self.size))
            member.train(train_loader, num_epochs, validation_loader)

    def add_metrics(self, metrics_list):
        for metric in metrics_list:
            if isinstance(metric, metrics.Metric):
                for member in self.members:
                    member.metrics[metric.name] = metric
                    self._log.info("Adding metric: {}".format(metric.name))
            else:
                self._log.error(
                    "Metric {} does not inherit from metric.Metric.".format(
                        metric.name))

    def calc_metrics(self, data_loader):
        for member in self.members:
            member.calc_metrics(data_loader)

    def get_logits(self, inputs):
        """Ensemble logits
        Returns the logits of all individual ensemble members.
        B = batch size, K = num output params, N = ensemble size

        Args:
            inputs (torch.tensor((B, data_dim))): data batch

        Returns:
            logits (torch.tensor((B, N, K)))
        """

        batch_size = inputs.size(0)
        logits = torch.zeros((batch_size, self.size, self.output_size),
                             device=self.device)
        for member_ind, member in enumerate(self.members):
            logits[:, member_ind, :] = member.forward(inputs)

        return logits

    def transform_logits(self, logits, transformation=None):
        """Ensemble predictions from logits
        Returns the predictions of all individual ensemble members,
        by applying the logits 'transformation' to the logits.
        B = batch size, K = num output params, N = ensemble size

        Args:
            transformed_logits (torch.tensor((B, N, K))): data batch
            transformation (funcion): maps logits to output space

        Returns:
            predictions (torch.tensor((B, N, K)))
        """

        batch_size = logits.size(0)
        transformed_logits = torch.zeros(
            (batch_size, self.size, self.output_size))
        for member_ind, member in enumerate(self.members):
            if transformation:
                transformed_logits[:, member_ind, :] = transformation(
                    logits[:, member_ind, :])
            else:
                transformed_logits[:, member_ind, :] = member.transform_logits(
                    logits[:, member_ind, :])

        return transformed_logits

    def predict(self, inputs, t=None):
        """Ensemble prediction
        Returns the predictions of all individual ensemble members.
        The return is actually a tuple with (pred_mean, all_predictions)
        for backwards compatibility but this should be removed.
        B = batch size, K = num output params, N = ensemble size
        TODO: Remove pred_mean and let the
        distilled model chose what to do with the output

        Args:
            inputs (torch.tensor((B, data_dim))): data batch

        Returns:
            predictions (torch.tensor((B, N, K)))
        """
        batch_size = inputs.size(0)
        predictions = torch.zeros((batch_size, self.size, self.output_size))
        for member_ind, member in enumerate(self.members):
            if t is None:
                predictions[:, member_ind, :] = member.predict(inputs)
            else:
                predictions[:, member_ind, :] = member.predict(inputs, t)

        return predictions

    def save_ensemble(self, filepath):

        members_dict = {}
        for i, member in enumerate(self.members):
            members_dict["ensemble_member_{}".format(i)] = member
            # To save memory one should save model.state_dict,
            # but then we also need to save class-type etc.,
            # so I will keep it like this for now

        torch.save(members_dict, filepath)

    def load_ensemble(self, filepath, num_members=None):

        check_point = torch.load(filepath)

        for i, key in enumerate(check_point):

            if num_members is not None and i == num_members:
                break

            member = check_point[key]
            # member.eval(), should be called if we have dropout or batch-norm
            # in our layers, to make sure that self.train = False,
            # just that it doesn't work for now
            self.add_member(member)


class EnsembleMember(nn.Module, ABC):
    """Parent class for keeping common logic in one place
    Instance variables:
        output_size (int): Represents the actual output size
            i.e number of dimensions D, or number of classes K
    """
    def __init__(self,
                 output_size,
                 loss_function,
                 target_size=None,
                 device=torch.device("cpu"),
                 grad_norm_bound=None):
        super().__init__()

        self._log = logging.getLogger(self.__class__.__name__)

        self.output_size = output_size
        if target_size is None:
            self.target_size = self.output_size
        else:
            self.target_size = target_size

        self.loss = loss_function
        self.metrics = dict()
        self.optimizer = None
        self.grad_norm_bound = grad_norm_bound
        self._log.info("Moving model to device: {}".format(device))
        self.device = device

        if self.loss is None:  # or not issubclass(type(self.loss),
            # nn.modules.loss._Loss): THIS DOES NOT WORK OUT

            raise ValueError("Must assign proper loss function to child.loss.")

    def train(self,
              train_loader,
              num_epochs,
              validation_loader=None,
              metrics=list(),
              reshape_targets=True):
        """Common train method for all ensemble member classes
        Should NOT be overridden!
        """
        store_loss = {"Train": list(), "Validation": list()}

        scheduler = torch_optim.lr_scheduler.StepLR(self.optimizer,
                                                    step_size=1,
                                                    gamma=0.1)

        # clr = utils.adapted_lr(c=0.7)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(
        #    self.optimizer, [clr])

        for epoch_number in range(1, num_epochs + 1):
            loss = self._train_epoch(train_loader, validation_loader, reshape_targets=reshape_targets)
            self._print_epoch(epoch_number, loss, "Train")
            store_loss["Train"].append(loss)
            if validation_loader is not None:
                loss = self._validate_epoch(validation_loader, reshape_targets=reshape_targets)
                self._print_epoch(epoch_number, loss, "Validation")
                store_loss["Validation"].append(loss)
            if self._learning_rate_condition(epoch_number):
                scheduler.step()
        return store_loss

    def _train_epoch(self,
                     train_loader,
                     validation_loader=None,
                     metrics=list(),
                     reshape_targets=True):
        """Common train epoch method for all ensemble member classes
        Should NOT be overridden!

        TODO: (Breaking change) Validation loader should be removed
        """

        self._reset_metrics()
        running_loss = 0.0
        for (batch_count, batch) in enumerate(train_loader):
            self.optimizer.zero_grad()
            inputs, targets = batch

            inputs, targets = inputs.float().to(
                self.device), targets.float().to(self.device)

            logits = self.forward(inputs)
            outputs = self.transform_logits(logits)

            if reshape_targets:
                # num_samples is different from batch size,
                # the loss expects a target with shape
                # (B, N, D), so that it can handle a full ensemble pred.
                # Here, we use a single sample N = 1.
                num_samples = 1
                batch_size = targets.size(0)
                targets = targets.reshape(
                    (batch_size, num_samples, self.output_size))

            loss = self.calculate_loss(outputs, targets)
            loss.backward()
            if self.grad_norm_bound is not None:
                nn.utils.clip_grad_norm(self.parameters(),
                                        self.grad_norm_bound)
            self.optimizer.step()
            running_loss += loss.item()

            self._update_metrics(outputs, targets)

        return running_loss / (batch_count + 1)

    def _validate_epoch(self, validation_loader, reshape_targets=True):
        """Common validate epoch method for all ensemble member classes
        Should NOT be overridden!
        """

        with torch.no_grad():
            self._reset_metrics()
            running_loss = 0.0
            batch_count = 0
            for batch in validation_loader:
                batch_count += 1

                inputs, targets = batch
                inputs, targets = inputs.float(),\
                    targets.float()
                logits = self.forward(inputs)
                outputs = self.transform_logits(logits)

                num_samples = 1
                batch_size = targets.size(0)

                if reshape_targets:
                    targets = targets.reshape(
                        (batch_size, num_samples, self.target_size))

                tmp_loss = self.calculate_loss(outputs, targets)
                running_loss += tmp_loss
                self._update_metrics(outputs, targets)

            return running_loss / batch_count

    def test(self, test_loader, metrics, loss_function):
        """Common test method for all ensemble member classes
        Should NOT be overridden!
        """

        with torch.no_grad():
            running_loss = 0.0
            batch_count = 0
            for batch in test_loader:
                batch_count += 1

                inputs, targets = batch
                inputs, targets = inputs.float().to(self.device),\
                    targets.float().to(self.device)
                logits = self.forward(inputs)
                outputs = self.transform_logits(logits)

                num_samples = 1
                batch_size = targets.size(0)
                targets = targets.reshape(
                    (batch_size, num_samples, self.target_size))

                tmp_loss = loss_function(outputs, targets)
                running_loss += tmp_loss
                for metric in metrics:
                    metric_output = self._output_to_metric_domain(
                        metric, outputs)
                    metric.update(targets, metric_output)

            return metrics, running_loss / batch_count

    def calc_metrics(self, data_loader):
        """Calculate all metrics"""
        self._reset_metrics()

        for batch in data_loader:
            inputs, targets = batch
            logits = self.forward(inputs)
            outputs = self.transform_logits(logits)
            self._update_metrics(outputs, targets)

        metric_string = ""
        for metric in self.metrics.values():
            metric_string += " {}".format(metric)
        self._log.info(metric_string)

    def _add_metric(self, metric):
        self.metrics[metric.name] = metric

    def _output_to_metric_domain(self, metric, outputs):
        """Transform output for metric calculation
        Output distribution parameters are not necessarily
        exact representation for metrics calculation.
        This helper function can be overloaded to massage the output
        into the correct shape

        If not overridden, it works as an identity map.
        """
        return outputs

    def _update_metrics(self, outputs, targets):
        for metric in self.metrics.values():
            metric_output = self._output_to_metric_domain(metric, outputs)
            metric.update(targets=targets, outputs=metric_output)

    def _reset_metrics(self):
        for metric in self.metrics.values():
            metric.reset()

    def _print_epoch(self, epoch_number, loss, type_="Train"):
        epoch_string = "{} - Epoch {}: Loss: {:.3f}".format(
            type_, epoch_number, loss)
        for metric in self.metrics.values():
            epoch_string += " {}".format(metric)
        self._log.info(epoch_string)

    def _learning_rate_condition(self, epoch):
        """Evaluate condition for increasing learning rate
        Defaults to never increasing. I.e. returns False
        """

        return False

    @abstractmethod
    def forward(self, inputs):
        """Forward method only produces logits.
        I.e. no softmax or other det. transformation.
        That is instead handled by transform_logits
        This is for flexibility when using the ensemble as teacher.
        """

    @abstractmethod
    def transform_logits(self, logits):
        """Transforms the networks logits
        (produced by the forward method)
        to a suitable output value, i.e. a softmax
        to generate a probability distr.

        Default impl. is not given to avoid this transf.
        being implicitly included in the forward method.

        Args:
            logits (torch.tensor(B, K)):
        """

    @abstractmethod
    def calculate_loss(self, targets, outputs):
        """Calculates loss"""

"""Distilled net base module"""
import logging
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.optim as torch_optim
import math
import src.utils as utils


class DistilledNet(nn.Module, ABC):
    """Parent class for distilled net logic in one place"""
    def __init__(self, teacher, loss_function, device=torch.device("cpu")):
        super().__init__()
        self._log = logging.getLogger(self.__class__.__name__)
        self.teacher = teacher
        self.loss = loss_function
        self.metrics = dict()
        self.use_hard_labels = False

        if self.loss is None or not issubclass(type(self.loss),
                                               nn.modules.loss._Loss):

            self._log.warning(
                "Must assign proper loss function to child.loss.")
        self.optimizer = None
        self._log.info("Moving model to device: {}".format(device))
        self.device = device

    def train(self, train_loader, num_epochs, validation_loader=None):
        """ Common train method for all distilled networks
        Should NOT be overridden!
        """

        clr = utils.adapted_lr(c=0.7)
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, [clr])

        self.use_hard_labels = False

        self._log.info("Training distilled network.")

        # Want to look at metrices at initialization
        if validation_loader is None:
            self.calculate_metric_dataloader(train_loader)
        else:
            self.calculate_metric_dataloader(validation_loader)
        
        for epoch_number in range(1, num_epochs + 1):

            loss = self._train_epoch(train_loader,
                                     validation_loader=validation_loader)
            self._print_epoch(epoch_number, loss)
            if self._learning_rate_condition(epoch_number):
                scheduler.step()

            if self._temp_annealing_schedule(epoch_number):
                self._temperature_anneling()

            if epoch_number % 10 == 0:
                acc = self.eval(validation_loader)
                self._log.info("Test accuracy is {}".format(acc))

        self._reset_metrics()  # For storing purposes

    def _train_epoch(self,
                     train_loader,
                     validation_loader=None):
        """Common train epoch method for all distilled networks
        Should NOT be overridden!
        TODO: Make sure train_loader returns None for labels,
        if no labels are available.
        """
        running_loss = 0

        self._reset_metrics()

        for batch_ind, batch in enumerate(train_loader):
            self.optimizer.zero_grad()
            inputs, labels = batch

            # Need this for a special case
            if isinstance(inputs, list):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].to(self.device)

                labels = labels.to(self.device)

            else:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

            teacher_predictions = self._generate_teacher_predictions(inputs)
            
            outputs = self.forward(inputs)
            
            loss = self.calculate_loss(outputs, teacher_predictions, None)

            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            self._log.debug("Batch: {}, running_loss: {}".format(
                batch_ind, running_loss))

            if math.isnan(running_loss) or math.isinf(running_loss):
                self._log.error("Loss is NaN")
                break

            if validation_loader is None:
                #self._reset_metrics()
                self._update_metrics(
                    outputs, teacher_predictions
                )

        if validation_loader is not None:
            with torch.no_grad():
                self.calculate_metric_dataloader(validation_loader)

        return running_loss

    def calculate_metric_dataloader(self, data_loader):
        for batch in data_loader:
            # self._reset_metrics()
            inputs, labels = batch
            
            if isinstance(inputs, list):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].to(self.device)

                labels = labels.to(self.device)

            else:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            outputs = self.forward(inputs)
            teacher_predictions = self._generate_teacher_predictions(inputs)
            teacher_predictions = teacher_predictions
            self._update_metrics(outputs, teacher_predictions, labels)

    def _generate_teacher_predictions(self, inputs):
        """Generate teacher predictions
        The intention is to get the logits of the ensemble members
        and then apply some transformation to get the desired predictions.
        Default implementation is to recreate the exact ensemble member output.
        Override this method if another logit transformation is desired,
        e.g. unit transformation if desired predictions
        are the logits themselves
        """

        logits = self.teacher.get_logits(inputs)
        return self.teacher.transform_logits(logits)

    def calc_metrics(
        self, data_loader
    ):  #TODO: How does this differ from calc_metric_dataloader except for the reset_metrics call?
        self._reset_metrics()

        for batch in data_loader:
            inputs, targets = batch
            
            outputs = self.forward(inputs)
            teacher_predictions = self._generate_teacher_predictions(inputs)
            self._update_metrics(outputs, teacher_predictions)

        metric_string = ""
        for metric in self.metrics.values():
            metric_string += " {}".format(metric)
        self._log.info(metric_string)


    # TODO: REMOVE?
    def get_scheduler(self, step_size=100, factor=100000, cyclical=False):

        if cyclical:
            end_lr = self.learning_rate
            clr = utils.cyclical_lr(step_size,
                                    min_lr=end_lr / factor,
                                    max_lr=end_lr)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, [clr])
        else:
            scheduler = torch_optim.lr_scheduler.StepLR(self.optimizer,
                                                        step_size=step_size,
                                                        gamma=0.9)

        return scheduler

    def add_metric(self, metric):
        self.metrics[metric.name] = metric

    def _update_metrics(self, outputs, labels, true_labels):
        for metric in self.metrics.values():
            metric.update(targets=labels, outputs=outputs, true_labels = true_labels)

    def _reset_metrics(self):
        for metric in self.metrics.values():
            metric.reset()

    def _print_epoch(self, epoch_number, loss):
        epoch_string = "Epoch {}: Loss: {}".format(epoch_number, loss)
        for metric in self.metrics.values():
            epoch_string += " {}".format(metric)
        self._log.info(epoch_string)

    def _learning_rate_condition(self, epoch=None):
        """Evaluate condition for increasing learning rate
        Defaults to never increasing. I.e. returns False
        """
        return False

    def _temp_annealing_schedule(self, epoch=None):
        """Evaluate condition for softmax temperature annealing"""
        return False

    @abstractmethod
    def forward(self, inputs):
        pass

    @abstractmethod
    def calculate_loss(self, outputs, teacher_predictions, labels=None):
        pass

    @abstractmethod
    def calculate_acc(self, outputs, teacher_predictions, labels=None):
        pass
    
    @abstractmethod
    def _temperature_anneling(self, temp_factor=0.95):
        pass

    @abstractmethod
    def eval(self, loader):
        pass
    
    
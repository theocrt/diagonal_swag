import copy
import torch
import numpy as np

class SWAG:
    def __init__(self, model):
        """ 
        Initialize Stochastic Weights Averaging Gaussian for a PyTorch model.
        For now, only the SWAG-Diagonal is implemented. See A Simple Baseline for Bayesian Uncertainty
        in Deep Learning, Maddox 2019
        
        Parameters
        -------------------------------------
            model : pretrained PyTorch NN on which you want to use SWAG
        """

        self.model = copy.deepcopy(model)
        # Initial weights and squared weights
        theta0 = copy.deepcopy(model.state_dict())
        theta_sq0 = self.compute_theta_sq(theta0)

        # Keep track of theta_i and theta_i^2
        self.theta_list = {key: [param.clone().detach()] for key, param in theta0.items()}
        self.theta_sq_list = {key: [param.clone().detach() ** 2] for key, param in theta_sq0.items()}

    def compute_theta_sq(self, theta):
        """ Compute theta^2"""
        return {key: param ** 2 for key, param in theta.items()}

    def compute_SWA(self, dataloader, optimizer, criterion, T):
        """ 
        Performs SWA by running SGD iterates and averaging the weights

        Parameters
        -------------------------------------
            dataloader : PyTorch dataloader on which to performs the SGD iterates
            optimizer : (subject to change) PyTorch optimizer, should be SGD for the method to be theoritically valid
            criterion : PyTorch loss, to minimize with the SGD iterates
            T : int, number of SGD iteration to performs. (Warning : In this implementation every NN are stored which can be memory hungry)
            """
        
        i = 0
        # Collecting SGD iterates
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            predictions = self.model(x_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

            theta_i = self.model.state_dict()
            for key in self.theta_list.keys():
                self.theta_list[key].append(theta_i[key].clone().detach())
                self.theta_sq_list[key].append(theta_i[key].clone().detach() ** 2)

            if i >= T:
                break
            i += 1

        # Compute SWA mean and variance
        self.sigma_diag = {}
        self.theta_SWA = {}
        for key in theta_i.keys():
            theta_tensor = torch.stack(self.theta_list[key])
            theta_sq_tensor = torch.stack(self.theta_sq_list[key])

            theta_SWA = torch.mean(theta_tensor, dim=0)
            theta_sq_mean = torch.mean(theta_sq_tensor, dim=0)

            self.sigma_diag[key] = torch.clamp(theta_sq_mean - theta_SWA ** 2, min=1e-6)
            self.theta_SWA[key] = theta_SWA

    def sample_models(self, x, S):
        """ 
        Sample different NN from the posterior and predict on input point. The method compute_SWA 
        should be executed before using it.

        Parameters
        -------------------------------------
            x : torch.tensor, input on which you want to make a prediction
            S : int, number of NN sampled, i.e, number of predictions on x

        Returns
        -------------------------------------
            means : torch.tensor, mean of predictions on x of the differents NN
            std : torch.tensor, standard deviation of predictions on x
            preds : torhc.tensor, all predictions
        """
        preds = []
        for _ in range(S):
            theta_j = {}
            model_copy = copy.deepcopy(self.model)

            for key in self.model.state_dict().keys():
                std = torch.sqrt(self.sigma_diag[key])
                noise = torch.randn_like(std)
                theta_j[key] = self.theta_SWA[key] + std * noise

            with torch.no_grad():
                model_copy.load_state_dict(theta_j)
                pred_j = model_copy(x)
            preds.append(pred_j)

        preds = torch.stack(preds)
        return preds.mean(dim=0), preds.std(dim=0), preds

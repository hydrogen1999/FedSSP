import torch
import numpy as np
import random
import networkx as nx
from dtaidistance import dtw
from client import clientAvgSSP
import copy

class Server():
    def __init__(self, model, device):
        self.model = model.to(device)
        self.W = {key: value for key, value in self.model.named_parameters()}
        self.model_cache = []
        self.customized_params = {}

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_encoders = []
        self.uploaded_filters = []
        self.uploaded_feature_extractor = []

        self.clients = []
        self.selected_clients = []

        self.uploaded_model_gs = []

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.Budget = []


    def randomSample_clients(self, all_clients, frac):
        return random.sample(all_clients, int(len(all_clients) * frac))

    def receive_models_SSP(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in self.clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.model.base)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters_SSP(self):
        if not self.uploaded_models or not self.uploaded_weights:
            raise ValueError("No models or weights uploaded from clients. Ensure at least one client is active.")

        for model in self.uploaded_models:
            server_params = [name for name, _ in self.global_model.named_parameters()]
            client_params = [name for name, _ in model.named_parameters()]
            if server_params != client_params:
                raise ValueError("Inconsistent model structure between server and client.")
        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
        
        total_samples = sum(self.uploaded_weights)
        normalized_weights = [w / total_samples for w in self.uploaded_weights]

        for w, client_model in zip(normalized_weights, self.uploaded_models):
            for (server_name, server_param), (client_name, client_param) in zip(self.global_model.named_parameters(), client_model.named_parameters()):
                if 'encoder' in server_name:
                    if any(key in server_name for key in ['encoder', 'phi_e']):
                    server_param.data += client_param.data.clone() * w

    def send_models_SSP(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            client.set_parameters_SSP(self.global_model)


def flatten(source):
    return torch.cat([value.flatten() for value in source.values()])



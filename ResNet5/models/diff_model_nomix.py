import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffloss import DiffLoss

class DiffModel(nn.Module):
    def __init__(self, model, classifier=None, finetune_classifier=False, feature_dim=1024, target_dim=1024, diffloss_d=3, diffloss_w=1024,  num_sampling_steps='100', diffusion_batch_mul=50, grad_checkpointing=False, stage="stage-1", smooth=0.8, beta=1.0):
        super(DiffModel, self).__init__()
        self.model = model
        self.feature_dim = feature_dim
        self.target_dim = target_dim
        self.classifier = classifier
        self.stage = stage
        self.smooth = smooth
        self.beta = beta

        if stage == 'stage-2':
            # Diffusion Loss
            self.diffloss = DiffLoss(
                target_channels=self.target_dim,
                z_channels=self.feature_dim,
                width=diffloss_w,
                depth=diffloss_d,
                num_sampling_steps=num_sampling_steps,
                grad_checkpointing=grad_checkpointing
            )
            self.diffusion_batch_mul = diffusion_batch_mul
            self.ln = nn.LayerNorm(self.feature_dim, elementwise_affine=True)

            # cls-wise weight
            cls_centers = self.classifier.weight # 100 x 256
            cls_weight = F.softmax(cls_centers, dim=0)
            self.register_buffer('cls_centers', cls_centers)
            self.register_buffer('cls_weight', cls_weight)

            # finetune classifier
            self.finetune_classifier = finetune_classifier
            if not finetune_classifier:
                for param in self.classifier.parameters():
                    param.requires_grad = False
            else:
                for param in self.parameters():
                    param.requires_grad = False
                for param in self.classifier.parameters():
                    param.requires_grad = True
        elif stage == 'stage-1':
            for param in self.model.fc.parameters():
                param.requires_grad = False
            for param in self.classifier.parameters():
                param.requires_grad = False
            if feature_dim != target_dim:
                self.cls_projector = nn.Linear(feature_dim, target_dim)

    def forward_train_stage2(self, x, y, target_feature, weight=None):
        feats, _ = self.model(x, return_features=True)
        bsz, _ = target_feature.shape

        z = self.ln(feats[-1])
        logits = self.model.fc(z.detach())

        with torch.no_grad():
            centers = self.cls_centers[y]
            target_feature = target_feature * (1-self.smooth) + centers * self.smooth
            #if mask is not None:
            #    mask = mask.float()
            #    target_feature = target_feature * mask.unsqueeze(-1) + (1 - mask.unsqueeze(-1)) * centers  #Nx256 
        
        #indices = torch.randperm(z.size(0))
        #m = np.random.beta(self.beta, self.beta)
        #mix_z = z * m + (1-m) * z[indices]
        #mix_target_feature = target_feature * m + (1-m) * target_feature[indices]

        target_feature = target_feature.reshape(bsz, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bsz, -1).repeat(self.diffusion_batch_mul, 1)
        #mask = mask.reshape(bsz, -1).repeat(self.diffusion_batch_mul, 1)
        loss = self.diffloss(z=z, target=target_feature, weight=None)
        return loss, logits

    def forward_train_stage1(self, x):
        feats, _ = self.model(x, return_features=True)
        if self.feature_dim != self.target_dim:
            logits = self.classifier(self.cls_projector(feats[-1]))
        else:
            logits = self.classifier(feats[-1])
        return torch.Tensor([0]).to(logits), logits

    def forward_finetune_classifier(self, x):
        with torch.no_grad():
            feats, _= self.model(x, return_features=True)
            z = self.ln(feats[-1])
            sample_features = self.diffloss.sample(z, temperature=1.0)
        logits = self.classifier(sample_features)
        return torch.Tensor([0]).to(logits), logits

    def forward_test_stage1(self, x):
        feats, _ = self.model(x, return_features=True)
        if self.feature_dim != self.target_dim:
            logits = self.classifier(self.cls_projector(feats[-1]))
        else:
            logits = self.classifier(feats[-1])
        return logits

    def forward_test_stage2(self, x):
        feats, _ = self.model(x, return_features=True)
        z = self.ln(feats[-1])
        logits_new = self.model.fc(z) 

        sample_features = self.diffloss.sample(z, temperature=1.0)
        logits = self.classifier(sample_features)
        # multiple sampling
        #for i in range(9):
        #    sample_features = self.diffloss.sample(z, temperature=1.0)
        #    logits += self.classifier(sample_features)
        #logits = logits / 10.0
        return [logits, logits_new]

    def forward(self, x, y=None, target_feature=None, weight=None):
        if self.training and self.stage == 'stage-1':
            return self.forward_train_stage1(x)
        elif self.training and self.stage == 'stage-2' and (not self.finetune_classifier):
            return self.forward_train_stage2(x, y, target_feature, weight)
        elif self.training and self.stage == 'stage-2' and self.finetune_classifier:
            return self.forward_finetune_classifier(x)
        elif not self.training and self.stage == 'stage-1':
            return self.forward_test_stage1(x)
        elif not self.training and self.stage == 'stage-2':
            return self.forward_test_stage2(x)

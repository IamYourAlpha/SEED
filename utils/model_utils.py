import os
import torch
import models.cifar as models
import torch.optim as optim
from .ms_net_utils import *
from .data_utils import *


def load_teacher_network():
    """ return the best teacher network with state_dict. """

    teacher = models.__dict__['resnext'](
                    cardinality=8,
                    num_classes=10,
                    depth=29,
                    widen_factor=4,
                    dropRate=0,
                )
    teacher = torch.nn.DataParallel(teacher).cuda()
    checkpoint = torch.load("./ck_backup/teachers/resnext_best.pth.tar")
    teacher.load_state_dict(checkpoint['state_dict'])
    return teacher

def load_state_dict_for_experts(lois, # List Of IndexeS (lois)
                                experts,
                                dataset, 
                                arch, 
                                depth,
                                first_init=True,
                                path='./checkpoint_experts/wts/'
                                ):
    if (path is None):
        print ("Path is none first time")
        chk = torch.load('./ck_backup/%s/%s-depth-%s/checkpoint/model_best.pth.tar'%(dataset, arch, depth))
        for loi in lois:
            experts[loi].load_state_dict(chk['state_dict'])
    else:
        if (first_init):
            list_of_all_wts = os.listdir(path)
            for wt, loi in zip(list_of_all_wts, lois):
                wt_full_path = os.path.join(path, wt)
                wt_file = torch.load(wt_full_path)
                #experts[loi] = torch.nn.DataParallel(experts[loi]).cuda()
                experts[loi].load_state_dict(wt_file['state_dict'])
                experts[loi] = experts[loi].cuda()
                print ("load success for experts: {}".format(loi))
        else:
            for loi in lois:
                wts_loc = os.path.join(path, '%s'%loi + '.pth.tar')
                pth_exists = os.path.exists(wts_loc)
                if (pth_exists):
                    print("loading wts from: {}".format(wts_loc))
                    wts = torch.load(wts_loc)
                    experts[loi].load_state_dict(wts)
                else:
                    print ("passing")
    return experts
        
        
    
def load_expert_networks_and_optimizers(lois, 
                                        num_classes, 
                                        dataset, 
                                        arch, 
                                        depth, 
                                        block_name,
                                        initialize_with_router=True,
                                        finetune_experts=True
                                        ):
    experts = {}
    eoptimizers = {}
    chk = torch.load('./ck_backup/%s/%s-depth-%s/checkpoint/model_best.pth.tar'%(dataset, arch, depth))
    for loi in lois:
        experts[loi] = models.__dict__[arch](
                    num_classes=num_classes,
                    depth=depth,
                    block_name=block_name)
        
        experts[loi] = experts[loi].cuda()
        #experts[loi] = torch.nn.DataParallel(experts[loi]).cuda()
        
        # initialize_with_router = True
        # if (initialize_with_router):
        #     experts[loi].load_state_dict(chk['state_dict'])
        if (finetune_experts):
            # eoptimizers[loi] = optim.SGD([{'params': experts[loi].module.layer1.parameters(), 'lr': 0.001},
            #                             {'params': experts[loi].module.layer2.parameters(), 'lr': 0.001},
            #                              {'params': experts[loi].module.layer3.parameters(), 'lr': 0.01},
            #                              {'params': experts[loi].module.fc.parameters()}],
            #                              lr=0.01, momentum=0.9, weight_decay=5e-4)
            eoptimizers[loi] = optim.SGD([{'params': experts[loi].layer1.parameters(), 'lr': 0.0001},
                                        {'params': experts[loi].layer2.parameters(), 'lr': 0.0001},
                                         {'params': experts[loi].layer3.parameters(), 'lr': 0.001},
                                         {'params': experts[loi].fc.parameters()}],
                                         lr=0.01, momentum=0.9, weight_decay=5e-4)
            
        else:
            eoptimizers[loi] = optim.SGD(experts[loi].parameters(), lr=0.1, momentum=0.9,
                      weight_decay=5e-4)
            
        
    return experts, eoptimizers


def reset_optimizer(lois, experts):
    ''' resets the optimizer to initial learning rate '''
    
    eoptimizers = {}
    for loi in lois:
        eoptimizers[loi] = optim.SGD([{'params': experts[loi].layer1.parameters(), 'lr': 0.0001},
                                        {'params': experts[loi].layer2.parameters(), 'lr': 0.0001},
                                         {'params': experts[loi].layer3.parameters(), 'lr': 0.001},
                                         {'params': experts[loi].fc.parameters()}],
                                         lr=0.01, momentum=0.9, weight_decay=5e-4)
    return eoptimizers


def make_router_and_optimizer(num_classes,
                              dataset,
                              arch,
                              depth,
                              block_name,
                              learning_rate,
                              load_weights=False):
    model = models.__dict__[arch](
                    num_classes=num_classes,
                    depth=depth,
                    block_name=block_name)
    if (load_weights):
        #model = torch.nn.DataParallel(model).cuda()
        model = model.cuda()
        print ('./ck_backup/%s/%s-depth-%s/checkpoint/model_best.pth.tar'%(dataset, arch, depth))
        chk = torch.load('./ck_backup/%s/%s-depth-%s/checkpoint/model_best.pth.tar'%(dataset, arch, depth))
        model.load_state_dict(chk['state_dict'])
    
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9,
                      weight_decay=5e-4)
    return model, optimizer

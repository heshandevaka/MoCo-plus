from LibMTL.config import LibMTL_args
import sys, pytest
sys.path.append('../examples/office')
from train_office import main, parse_args

params = parse_args(LibMTL_args)
params.dataset = 'office-home'
params.dataset_path = '/'
params.epochs = 1
params.bs = 2
params.multi_input = True

def test_EW():
    params.rep_grad = False
    main(params)

def test_UW():
    params.rep_grad = False
    params.weighting = 'UW'
    main(params)

def test_RLW():
    params.rep_grad = False
    params.weighting = 'RLW'
    main(params)

def test_DWA():
    params.rep_grad = False
    params.weighting = 'DWA'
    params.epochs = 2
    main(params)
    params.epochs = 1

def test_GLS():
    params.rep_grad = False
    params.weighting = 'GLS'
    main(params)

def test_MGDA():
    for mgda_gn in ['none', 'l2', 'loss', 'loss+']:
        for rep_grad in [True, False]:
            params.rep_grad = rep_grad
            params.weighting = 'MGDA'
            params.mgda_gn = mgda_gn

            main(params)

def test_CAGrad():
    params.rep_grad = False
    params.weighting = 'CAGrad'
    main(params)

    with pytest.raises(ValueError):
        params.rep_grad = True
        params.weighting = 'CAGrad'
        main(params)

def test_GradDrop():
    params.rep_grad = True
    params.weighting = 'GradDrop'
    main(params)

    with pytest.raises(ValueError):
        params.rep_grad = False
        params.weighting = 'GradDrop'
        main(params)

def test_PCGrad():
    params.rep_grad = False
    params.weighting = 'PCGrad'
    main(params)

    with pytest.raises(ValueError):
        params.rep_grad = True
        params.weighting = 'PCGrad'
        main(params)

def test_Nash_MTL():
    params.rep_grad = False
    params.weighting = 'Nash_MTL'
    main(params)

    with pytest.raises(ValueError):
        params.rep_grad = True
        params.weighting = 'Nash_MTL'
        main(params)

def test_GradVac():
    params.rep_grad = False
    params.weighting = 'GradVac'
    main(params)

    with pytest.raises(ValueError):
        params.rep_grad = True
        params.weighting = 'GradVac'
        main(params)

def test_GradNorm():
    params.rep_grad = False
    params.weighting = 'GradNorm'
    params.epochs = 2
    main(params)
    params.epochs = 1

    params.rep_grad = True
    params.weighting = 'GradNorm'
    main(params)

def test_IMTL():
    params.rep_grad = False
    params.weighting = 'IMTL'
    main(params)

    params.rep_grad = True
    params.weighting = 'IMTL'
    main(params)


def test_MTAN():
    params.arch = 'MTAN'
    main(params)

def test_LTB():
    params.arch = 'LTB'
    main(params)

def test_Cross_stitch():
    with pytest.raises(ValueError):
        params.arch = 'Cross_stitch'
        main(params)

def test_CGC():
    params.arch = 'CGC'
    params.img_size = [3, 224, 224]
    params.num_experts = [1, 1, 1, 1]
    main(params)

def test_PLE():
    params.arch = 'PLE'
    params.img_size = [3, 224, 224]
    params.num_experts = [1, 1, 1, 1]
    main(params)

def test_MMoE():
    params.arch = 'MMoE'
    params.img_size = [3, 224, 224]
    params.num_experts = [2]
    main(params)

def test_DSelect_k():
    params.arch = 'DSelect_k'
    params.img_size = [3, 224, 224]
    params.num_experts = [2]
    main(params)
from GNN_workflow import FeatureConfig

Dealkylation_CONFIG = FeatureConfig(
    elem_list=[6,7,8,9,14,15,16,17,35,53],
    chirality=['CHI_UNSPECIFIED','CHI_TETRAHEDRAL_CW','CHI_TETRAHEDRAL_CCW'],
    degree=[1,2,3,4],
    hybridization=['SP','SP2','SP3'],
    bond_types=["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"],
    stereo=['STEREONONE','STEREOZ','STEREOE']
)

GlutathioneConjugation_CONFIG = FeatureConfig(
    elem_list=[6,7,8,9,15,16,17,35,53],
    chirality=['CHI_UNSPECIFIED','CHI_TETRAHEDRAL_CW','CHI_TETRAHEDRAL_CCW'],
    degree=[1,2,3,4],
    hybridization=['SP','SP2','SP3'],
    bond_types=["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"],
    stereo=['STEREONONE','STEREOZ','STEREOE']
)

Glucuronidation_CONFIG = FeatureConfig(
    elem_list=[6,7,8,9,14,15,16,17,35,53],
    chirality=['CHI_UNSPECIFIED','CHI_TETRAHEDRAL_CW','CHI_TETRAHEDRAL_CCW'],
    degree=[1,2,3,4],
    hybridization=['SP','SP2','SP3'],
    bond_types=["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"],
    stereo=['STEREONONE','STEREOZ','STEREOE']
)

Hydrolysis_CONFIG = FeatureConfig(
    elem_list=[6,7,8,9,15,16,17,35,53],
    chirality=['CHI_UNSPECIFIED','CHI_TETRAHEDRAL_CW','CHI_TETRAHEDRAL_CCW'],
    degree=[1,2,3,4],
    hybridization=['SP','SP2','SP3'],
    bond_types=["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"],
    stereo=['STEREONONE','STEREOZ','STEREOE']
)

Reduction_CONFIG = FeatureConfig(
    elem_list=[6,7,8,9,14,15,16,17,35,53],
    chirality=['CHI_UNSPECIFIED','CHI_TETRAHEDRAL_CW','CHI_TETRAHEDRAL_CCW'],
    degree=[1,2,3,4],
    hybridization=['SP','SP2','SP3'],
    bond_types=["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"],
    stereo=['STEREONONE','STEREOZ','STEREOE']
)

Oxidation_CONFIG = FeatureConfig(
    elem_list=[6,7,8,9,14,15,16,17,35,53],
    chirality=['CHI_UNSPECIFIED','CHI_TETRAHEDRAL_CW','CHI_TETRAHEDRAL_CCW'],
    degree=[1,2,3,4],
    hybridization=['SP','SP2','SP3'],
    bond_types=["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"],
    stereo=['STEREONONE','STEREOZ','STEREOE']
)

Sulfonation_CONFIG = FeatureConfig(
    elem_list=[6,7,8,9,14,16,17,35,53],
    chirality=['CHI_UNSPECIFIED','CHI_TETRAHEDRAL_CW','CHI_TETRAHEDRAL_CCW'],
    degree=[1,2,3,4],
    hybridization=['SP','SP2','SP3'],
    bond_types=["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"],
    stereo=['STEREONONE','STEREOZ','STEREOE']
)
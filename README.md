# intelligent_antibodies


## Get the data

All the data that can be used fir the challenge can be found on [SabDab](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab)

To get access to all the data the [search module]() is used. 

To get data containing both antibody and proteic antigene sequences with affinity use [this](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/search/?ABtype=All&method=All&species=All&resolution=&rfactor=&antigen=Protein&ltype=All&constantregion=All&affinity=True&chothiapos=&restype=ALA) - 468 entries
To get data containing both antibody and proteic antigene sequences without affinity use [this](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/search/?ABtype=All&method=All&species=All&resolution=&rfactor=&antigen=Protein&ltype=All&constantregion=All&affinity=All&chothiapos=&restype=ALA) - 5092 entries.

To get data containing both antibody and non necessary proteic antigene sequences with affinity use [this](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/search/?ABtype=All&method=All&species=All&resolution=&rfactor=&antigen=All&ltype=All&constantregion=All&affinity=True&chothiapos=&restype=ALA) - 737 entries

To get data containing both antibody and non necessary proteic antigene sequences without affinity use [this](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/search/?ABtype=All&method=All&species=All&resolution=&rfactor=&antigen=All&ltype=All&constantregion=All&affinity=All&chothiapos=&restype=ALA) - 7825 entries.

More criteria can be applied to select data from [here](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/search/)




Backup data can be found [here](https://github.com/mit-ll/AlphaSeq_Antibody_Dataset)

## Covid data

[Here](https://opig.stats.ox.ac.uk/webapps/covabdab/)

# Resources

[Article](https://www.sciencedirect.com/science/article/pii/S1093326322002431)


[Benchmark](https://github.com/piercelab/antibody_benchmark)


[Siamese Network](https://github.com/emersON106/AbAgIntPre)
https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2022.1053617/full#h6x

# Data
https://github.com/emersON106/AbAgIntPre/tree/main

 - Get this [data](https://github.com/emersON106/AbAgIntPre/tree/main/SAbDab) to have all the usefull PDBs, then collect all the corresponding fasta files.
 - Parse all the Fasta grepping "heavy chain", "light chain", "antibody", "antigene" to create dataset of sequences for both Ag and Ab.

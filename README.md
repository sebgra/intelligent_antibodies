# Intelligent Antibodies

Therapeutic (monoclonal) antibodies are one of the most effective therapies available today for the treatment of chronic inflammatory diseases such as Crohn's disease, lupus and multiple sclerosis. To treat the latter, monoclonal antibodies can target certain proteins involved in these pathologies with a view to neutralizing them, and can also be used to limit the supply of factors essential to tumor growth or disruptors of the tumor microenvironment. Monoclonal antibody-based serotherapy can also compensate for treatment shortfalls in the case of fulminant epidemics where the pathogens involved have a high mutability rate, such as COVID-19.

Although promising and a major product on the pharmaceutical market, only around thirty monoclonal antibodies are currently available for chronic inflammatory diseases, and around ten for the treatment of cancer. This lack of comprehensiveness is due to the many difficulties inherent in the in-vitro and in-silico design of these therapeutic molecules. Antibody design and/or optimization remains a real challenge, not least because of the need to produce molecules that are effective, target-specific and deliverable to the organs being treated. The difficulties are also linked to long and costly development times.

In order to accelerate the development of therapeutic antibodies, in-silico methods have been developed to reduce modeling times for these molecules, while exploring design possibilities more exhaustively. Although advantageous, these methods currently rely essentially on estimating the affinity between the antibody and its target by calculating the binding energy, which remains difficult to estimate and extremely time-consuming from an experimental point of view.


# Data

Two data sets are available, one about multiple species from [SabDab](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab), the other about COVID [Cov-AbDab] [https://opig.stats.ox.ac.uk/webapps/covabdab/]. 

All data previously mentionned are free to acces. 

### SabDab dataset

The data use here are part of SabDab. They relates to _immune complexes_ characterized through X-ray crystallography. 

Two files have been collected :

- All_PDB_files.txt : which contains ids for each constitutive proteine-protein structure (Antigen-Antibody). The first four characters of the structure name refer to the [RCSB PDB](https://www.rcsb.org/) database. Second part of the ids concern the chains inside the structures. 

- Positive_samples.txt : which contains all the positively interacting proteins from same or different complexes. For instance, ```4gms_J_N_E	2vir_B_A_C``` relates interactions between _4gms_ and _2vir_. No specific order is precised, meaning that first partner can act the antigen or the antibody. This is reciprocal. Indedd as there are complexes, the antigenic chains of _4gms_ can form immune complexes with antibody chains of _2vir_ and antigenic chains of _2vir_ can also form immune complexes with antibody chains of _4gms_. Obviously antigenic and antibody parts chains of a given RCSB ids are forming complexes. 

Therefore, two complexes ids that are not matched are not able to form complexes and will act as negative samples. 

All the structures are directly collected from SabDab as fasta files looking as : 

```
>1A2Y_1|Chain A|IGG1-KAPPA D1.3 FV (LIGHT CHAIN)|Mus musculus (10090)
DIVLTQSPASLSASVGETVTITCRASGNIHNYLAWYQQKQGKSPQLLVYYTTTLADGVPSRFSGSGSGTQYSLKINSLQPEDFGSYYCQHFWSTPRTFGGGTKLEIK
>1A2Y_2|Chain B|IGG1-KAPPA D1.3 FV (HEAVY CHAIN)|Mus musculus (10090)
QVQLQESGPGLVAPSQSLSITCTVSGFSLTGYGVNWVRQPPGKGLEWLGMIWGDGNTDYNSALKSRLSISKDNSKSQVFLKMNSLHTDDTARYYCARERDYRLDYWGQGTTLTVSS
>1A2Y_3|Chain C|LYSOZYME|Gallus gallus (9031)
KVFGRCELAAAMKRHGLANYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL
```

Sequence informations lines start with _>_ and following line correspond to the constitutive chain of residues (amino-acids). 

### Cov-AbDab

The data use here are part of Cov-AbDab. They relates to _immune complexes_.

Three files are available : 

- positive dataset.txt : a three column file metionning the Sars-cov identifier, the antibody sequence and the antigen sequence. Such chains can form immune complexes.

- negative dataset.txt : a three column file metionning the Sars-cov identifier, the antibody sequence and the antigen sequence. Such chains do not  form immune complexes.

- Independant test.txt  : 


## Get the data

For convenience, some scripts have been written to parse SAbDab database to collect all the fasta files of complexes and to structures sequences table. 

To get all the sequences please use _./scripts/download_pdb.py_ through : 

```
python download_pdb.py
```

All fasta files will be saved in _./data/SabDab/fasta_ folder.

To get the interaction table use : 

```
python get_interaction_table.py
```

This will create _data.csv_ a three column tabular file containing antibody identifier, antigen identifier and 0/1 depending on the ability to form immune complexes. Identifiers are supplemented wit _|ag_ or _|ab_ to refer to the antigenic or antiboy part of the complex.

The data look like : 

```
ab;ag;interaction
5kel|ab;5kel|ag;1
5kel|ab;6cwt|ag;0
...
```

To have the sequences table use :

```
python get_seq_table.py
```

This will create _sequences_.csv_ a three column tabular file containing extended identifier such as _abcd|ag_ or _abcd|ab_, the species from where it comes and the chain of residues.

```
seq_id;specie;sequence
5kel|ag;Zaire ebolavirus (strain Mayinga-76) (128952);IPLGVIHNSTLQVSDVDKLVCRDKLSSTNQLRSVGLNLEGNGVATDVPSATKRWGFRSGVPPKVVNYEAGEWAENCYNLEIKKPDGSECLPAAPDGIRGFPRCRYVHKVSGTGPCAGDFAFHKEGAFFLYDRLASTVIYRGTTFAEGVVAFLILPQAKKDFFSSHPLREPVNATEDPSSGYYSTTIRYQATGFGTNETEYLFEVDNLTYVQLESRFTPQFLLQLNETIYTSGKRSNTTGKLIWKVNPEIDTTIGEWAFWETKKNLTRKIRSEELSFTVVSNGAKNISGQSPARTSSDPGTNTTTEDHKIMASENSSAMVQVHSQGREAAVSHLTTLATISTSPQSLTTKPGPDNSTHNTPVYKLDISEATQVEQHHRRTDNDSTASDTPSATTAAGPPKAENTNTSKSTDFLDPATTTSPQNHSETAGNNNTHHQDTGEESASSGKLGLITNTIAGVAGLITGGRRTRR
5kel|ag;Zaire ebolavirus (128952);EAIVNAQPKCNPNLHYWTTQDEGAAIGLAWIPYFGPAAEGIYTEGLMHNQDGLICGLRQLANETTQALQLFLRATTELRTFSILNRKAIDFLLQRWGGTCHILGPDCCIEPHDWTKNITDKIDQIIHDFVDKTLPDLEVDDDD
...
```

# Strategy

## Data encoding

https://dmnfarrell.github.io/bioinformatics/mhclearning

## Neural networks

## Draft

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

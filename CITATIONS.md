# daisybio/domainbenchmark: Citations

## [nf-core](https://pubmed.ncbi.nlm.nih.gov/32055031/)

> Ewels PA, Peltzer A, Fillinger S, Patel H, Alneberg J, Wilm A, Garcia MU, Di Tommaso P, Nahnsen S. The nf-core framework for community-curated bioinformatics pipelines. Nat Biotechnol. 2020 Mar;38(3):276-278. doi: 10.1038/s41587-020-0439-x. PubMed PMID: 32055031.

## [Nextflow](https://pubmed.ncbi.nlm.nih.gov/28398311/)

> Di Tommaso P, Chatzou M, Floden EW, Barja PP, Palumbo E, Notredame C. Nextflow enables reproducible computational workflows. Nat Biotechnol. 2017 Apr 11;35(4):316-319. doi: 10.1038/nbt.3820. PubMed PMID: 28398311.

## Pipeline tools

- [MultiQC](https://pubmed.ncbi.nlm.nih.gov/27312411/)

  > Ewels P, Magnusson M, Lundin S, Käller M. MultiQC: summarize analysis results for multiple tools and samples in a single report. Bioinformatics. 2016 Oct 1;32(19):3047-8. doi: 10.1093/bioinformatics/btw354. Epub 2016 Jun 16. PubMed PMID: 27312411; PubMed Central PMCID: PMC5039924.

- [scikit-learn](https://jmlr.org/papers/v12/pedregosa11a.html)

  > Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research. 2011;12:2825-2830.

- [PyTorch](https://papers.nips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library)

  > Paszke A, Gross S, Massa F, Lerer A, Bradbury J, Chanan G, et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library. Advances in Neural Information Processing Systems 32. 2019;8024-8035.

- [skorch](https://github.com/skorch-dev/skorch)

  > Tietz M, Fan TJ, Nouri D, Bossan B, et al. skorch: A scikit-learn compatible neural network library that wraps PyTorch. 2017.

- [RAPIDS cuML](https://docs.rapids.ai/api/cuml/stable/)

  > Raschka S, Patterson J, Nolet C. Machine Learning in Python: Main developments and technology trends in data science, machine learning, and artificial intelligence. Information. 2020;11(4):193. doi: 10.3390/info11040193. (Accelerated via NVIDIA RAPIDS cuML.)

- [pandas](https://pandas.pydata.org/)

  > McKinney W. Data Structures for Statistical Computing in Python. Proc 9th Python in Science Conference. 2010;56-61. doi: 10.25080/Majora-92bf1922-00a.

- [NumPy](https://www.nature.com/articles/s41586-020-2649-2)

  > Harris CR, Millman KJ, van der Walt SJ, et al. Array programming with NumPy. Nature. 2020;585:357-362. doi: 10.1038/s41586-020-2649-2.

- [SciPy](https://www.nature.com/articles/s41592-019-0686-2)

  > Virtanen P, Gommers R, Oliphant TE, et al. SciPy 1.0: fundamental algorithms for scientific computing in Python. Nat Methods. 2020;17:261-272. doi: 10.1038/s41592-019-0686-2.

- [h5py](https://www.h5py.org/)

  > Collette A. Python and HDF5. O'Reilly Media. 2013.

- [NetworkX](https://networkx.org/)

  > Hagberg AA, Schult DA, Swart PJ. Exploring network structure, dynamics, and function using NetworkX. Proc 7th Python in Science Conference. 2008;11-15.

- [GOATools](https://github.com/tanghaibao/goatools)

  > Klopfenstein DV, Zhang L, Pedersen BS, et al. GOATOOLS: A Python library for Gene Ontology analyses. Sci Rep. 2018;8:10872. doi: 10.1038/s41598-018-28948-z.

## Protein language models / embeddings

- [ESM-3](https://www.science.org/doi/10.1126/science.ads0018)

  > Hayes T, Rao R, Akin H, et al. Simulating 500 million years of evolution with a language model. Science. 2025;387(6736):850-858. doi: 10.1126/science.ads0018.

- [ESM Cambrian (ESM-C)](https://www.evolutionaryscale.ai/blog/esm-cambrian)

  > EvolutionaryScale Team. ESM Cambrian: Revealing the mysteries of proteins with unsupervised learning. EvolutionaryScale technical report. 2024.

- [ProtT5](https://ieeexplore.ieee.org/document/9477085)

  > Elnaggar A, Heinzinger M, Dallago C, et al. ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning. IEEE Trans Pattern Anal Mach Intell. 2022;44(10):7112-7127. doi: 10.1109/TPAMI.2021.3095381.

## Domain-domain interaction (DDI) sources

- [3did](https://3did.irbbarcelona.org/)

  > Mosca R, Céol A, Stein A, Olivella R, Aloy P. 3did: a catalog of domain-based interactions of known three-dimensional structure. Nucleic Acids Res. 2014 Jan;42(Database issue):D374-9. doi: 10.1093/nar/gkt887.

- [Pfam](https://www.ebi.ac.uk/interpro/)

  > Mistry J, Chuguransky S, Williams L, et al. Pfam: The protein families database in 2021. Nucleic Acids Res. 2021;49(D1):D412-D419. doi: 10.1093/nar/gkaa913.

- [STRING](https://string-db.org/)

  > Szklarczyk D, Kirsch R, Koutrouli M, et al. The STRING database in 2023. Nucleic Acids Res. 2023;51(D1):D638-D646. doi: 10.1093/nar/gkac1000.

## Graph / parsimony models reimplemented in this pipeline

- **KG-IDDI** (`kgiddi`)

  > Sherif AbouSheaisha N, Al-Athamneh A, Bertelli C, et al. Predicting domain-domain interactions using a parsimony approach over knowledge graphs. (Reimplementation; see `bin/kgiddi.py` for the algorithm and inline references.)

- **DDI parsimony** (`ddiparsimony`)

  > Riley R, Lee C, Sabatti C, Eisenberg D. Inferring protein domain interactions from databases of interacting proteins. Genome Biol. 2005;6(10):R89. doi: 10.1186/gb-2005-6-10-r89. (See `bin/ddiparsimony.py`.)

## Software packaging, distribution and reproducibility

- [Anaconda](https://anaconda.com)

  > Anaconda Software Distribution. Computer software. Vers. 2-2.4.0. Anaconda, Nov. 2016. Web.

- [Bioconda](https://bioconda.github.io/)

  > Grüning B, Dale R, Sjödin A, et al. Bioconda: sustainable and comprehensive software distribution for the life sciences. Nat Methods. 2018 Jul;15(7):475-476. doi: 10.1038/s41592-018-0046-7.

- [BioContainers](https://biocontainers.pro/)

  > da Veiga Leprevost F, Grüning B, Aflitos SA, et al. BioContainers: an open-source and community-driven framework for software standardization. Bioinformatics. 2017 Aug 15;33(16):2580-2582. doi: 10.1093/bioinformatics/btx192.

- [Docker](https://dl.acm.org/doi/10.5555/2600239.2600241)

  > Merkel D. Docker: lightweight Linux containers for consistent development and deployment. Linux J. 2014 Mar;2014(239):2.

- [Singularity / Apptainer](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0177459)

  > Kurtzer GM, Sochat V, Bauer MW. Singularity: Scientific containers for mobility of compute. PLoS One. 2017 May 11;12(5):e0177459. doi: 10.1371/journal.pone.0177459.

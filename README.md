# ATLAS Project
### Running the Notebooks
My analysis is split into two parts (as the notebooks were getting quite long). The first part (*Part1_DataProcessing*) involves reading the ATLAS data files, outputting a csv file (*DataFrame_Final.csv*) which contains extra columns for the Z-bosons, H-boson, angles and other information. The CSV file is not included in the repository, as it is ~700MB. The second part (*Part2_PlotsAnalysis*) involves reading in the CSV and making comparitive plots of different subsets of the data.

As far as I know, I did not add any extra dependencies besides that already included in your original repository. However, just in case, I've included a requirements file.

Additionally, you might need to edit the cell in which the file names are set, depending on how you've arranged your directories.

### A Note on Notation

Throughout the notebooks, I use the term 'L4' to refer to an arbitrary combination of 4 leptons, whereas 'llll' refers to the collective case in which we have 4 leptons of the same type, i.e., EEEE or UUUU.

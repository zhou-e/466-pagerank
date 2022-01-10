CSC 466 - Lab 6
Edward Zhou: ekzhou@calpoly.edu

Script: pageRank.py
---------------------------------
Instructions
There are 4 command line arguments expected: <filename> <d> <iterations flag> <iteration count/ epsilon>
<iterations flag> is either 0 or 1, where 0 inidcates pageRank.py will run until the Page Rank difference will be <= <iteration count/ epsilon>. 1 indicates pageRank.py will run the number of iterations in <iteration count/ epsilon>.

Notes:
- An assumption made by pageRank.py is the given file is an DIRECTED graph.
- pageRank.py only prints the top 10 rankings after the stopping condition specified, the full output of rankings are in <filename>_ranks.csv.
- The output of pageRank for the 7 chosen datasets can be found in the /output folder.
- Data files are in the /data and /snap-data folders.
- SNAP datasets are determined by or not the file is a .txt file (.csvs for non-SNAP datasets), other file types will NOT work.
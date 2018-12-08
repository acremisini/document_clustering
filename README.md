FILE PATHS: 
- In Globals.py, please set ECB_DIR to the directory where you've downloaded the ECB+ corpus. 
  -> ie., .../.../ECB+_LREC2014/ECB+
- Also in Globals.py, set ECB_SENTENCE_FILE exactly to the path where the included csv document is
  -> ie., .../.../ECB+_LREC2014/ECBplus_coreference_sentences.csv
NOTES:
- Running the code as is will perform all clustering experiments across the entire corpus, but not
  the word count or homonegenity metrics. 
- If you would like to see the word count/homonegenity metrics, we recommend that you set the "topics"
  parameter in the ecb_wrapper (line 14, pointing to the ECBWrapper object) to something like ['1','2'], 
  in order to compute the word count/homogeneity metrics in a reasonable amount of time. This will 
  not produce the same results as in the paper, but setting topics = None (the entire corpus) 
  will take about overnight to compute. Also, please set do_pairwise = True (line 30)
- The results of all procedures are written do the output directory uploaded to this repo. The results from the paper
  are included in this directory from our run on the entire dataset. 
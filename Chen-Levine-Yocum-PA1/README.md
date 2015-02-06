#COSI137B PA1
Aaron Levine, Tianzhi Chen, Zachary Yocum

While there are several scripts included in our submission, we've already produced the output from everything such that it is ready to be evaluated with the `evaluate-head.py` script:

    $ python evaluate-head.py test-ref.txt test-hyp.txt 
      0.784848484848
      0.704442429737
      p = 0.784848484848 r = 0.704442429737 f = 0.742474916388

##`prepdata.py`
We wrote this Python script to generate `train.crfsuite.txt`, `dev.crfsuite.txt`, and `test.crfsuite.txt`, which are formatted in CRFSuite's feature template format for model training and evaluation.  `train.crfsuite.txt` can be used to train a model as follows:

    $ crfsuite learn -m bio.model train.crfsuite.txt

##`bio.model`
This file is a CRFSuite model binary that can be used for tagging. E.g., you can tag the test data and write the hypothesized tag sequence by running the following:

    $ crfsuite tag -m bio.model test.crfsuite.txt > test-hyp.txt

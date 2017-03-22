import numpy as np

from nm_suppression import NMSuppression

boxes = np.array([(12, 84, 140, 212), (24, 84, 152, 212),
                  (36, 84, 164, 212), (12, 96, 140, 224),
                  (24, 96, 152, 224), (24, 108, 152, 236)])

f = NMSuppression(bbs= boxes, overlapThreshold= 0.5)
print f.fast_suppress()

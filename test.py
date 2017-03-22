from fast_nm_suppression import FNMSuppression

boxes = np.array([(12, 84, 140, 212), (24, 84, 152, 212),
                  (36, 84, 164, 212), (12, 96, 140, 224),
                  (24, 96, 152, 224), (24, 108, 152, 236)])

f = FNMSuppression(bbs= boxes, overlapThreshold= 0.5)
print f.suppress()

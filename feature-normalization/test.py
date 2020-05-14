from feature_normalize import FeatureProcessor, Z_Normalizer
zn = Z_Normalizer()
zn.fit("data/train.tsv")
pr = FeatureProcessor(zn)
pr.process("data/test.tsv", "data/test_proc.tsv")
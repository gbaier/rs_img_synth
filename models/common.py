from collections import namedtuple

# Named tuple for a neural networks output with the
# final output and intermediate features.
# Intermediate features are useful for computing FID
# or the discriminator feature loss of SPADE.
NNOutput = namedtuple("NNOutput", "final features")

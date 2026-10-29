# Pre-experiment Predictions

## Q2.5 - Before Testing Model A

Before running the mirror, rotation, or speed experiments, I expect model A to be less consistent after both image transformations, with rotation causing the larger failure because it changes digit appearance more strongly even though the transformed motion remains physically valid. The one-step reach is 8 original pixels, so I expect normal-speed prediction to remain usable, 2x motion to degrade, and 3x motion to fail once the displacement between observed frames exceeds what the recurrent update can connect.

## Q2.11 - Before Training Models B-D

I expect model B to reduce the mirror and especially rotation consistency failures, lowering the consistency gaps and transformed errors through symmetry augmentation, while leaving the fast-motion failure largely unchanged. Model C should improve the severe 2x and 3x errors by training on those speeds, but it may still fail at 3x if the 8-pixel one-step reach is the limiting factor. Model D should retain model C's speed adaptation and provide the strongest improvement at 3x because its larger reach can carry information across displacements that model A cannot connect.

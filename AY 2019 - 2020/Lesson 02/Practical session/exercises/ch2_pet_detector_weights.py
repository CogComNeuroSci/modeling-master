# setting up weights for the cat-dog detector
weights[0, 3] = .8 # cats often bite visitors
weights[0, 4] = .1 # dogs rarely bite visitors
weights[1, 3] = .2 # cats often have four legs
weights[1, 4] = .2 # dogs often have four legs
weights[2, 3] = .1 # cats rarely have their pictures on FB
weights[2, 4] = .8 # dogs often have their pictures on FB
weights[3, 4] = -.2 # a cat cannot be a dog, and vice versa